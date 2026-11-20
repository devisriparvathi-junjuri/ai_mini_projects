"""Unit tests for spam_classifier.py (standard library unittest, no extra packages).

Run all tests from the project folder with:

    python -m unittest -v
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import spam_classifier as sc


def capture(func, *args, **kwargs):
    """Call func and return (result, everything it printed)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def make_data(spam=6, ham=6):
    """Build a small balanced dataset with clearly different vocabulary per class."""
    data = [(f"winner claim your free cash prize now {i}", 1) for i in range(spam)]
    data += [(f"meeting notes for the project lunch tomorrow {i}", 0) for i in range(ham)]
    return data


def make_csv(spam=6, ham=6):
    """Same as make_data, but as the text of a CSV file with a header row."""
    lines = ["label,text"]
    lines += [f"{'spam' if label else 'ham'},{text}" for text, label in make_data(spam, ham)]
    return "\n".join(lines) + "\n"


class TempFileTestCase(unittest.TestCase):
    """Base class that gives each test a throwaway folder to write data files into."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = tmp.name

    def write_bytes(self, name, content):
        path = os.path.join(self.tmp_dir, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def write(self, name, content):
        # Encode ourselves so line endings are identical on Windows and Linux
        return self.write_bytes(name, content.encode("utf-8"))


class LoadDatasetTests(TempFileTestCase):
    def test_csv_with_header(self):
        path = self.write("a.csv", "label,text\nspam,Win a free prize\nham,See you at lunch\n")
        data, output = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Win a free prize", 1), ("See you at lunch", 0)])
        self.assertEqual(output, "")  # no warning when every row is valid

    def test_header_names_are_case_insensitive_and_any_order(self):
        path = self.write("a.csv", "Message,CATEGORY\nWin cash now,SPAM\nLunch at noon,Ham\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Win cash now", 1), ("Lunch at noon", 0)])

    def test_extra_columns_are_ignored(self):
        # Layout used by the popular Kaggle spam.csv: v1, v2 plus empty columns
        path = self.write("a.csv", "v1,v2,,,\nham,Hello there,,,\nspam,Free entry,,,\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Hello there", 0), ("Free entry", 1)])

    def test_numeric_labels(self):
        path = self.write("a.csv", "label,text\n1,Free prize\n0,See you soon\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Free prize", 1), ("See you soon", 0)])

    def test_headerless_tsv(self):
        path = self.write("a.tsv", "ham\tOk lar... Joking wif u oni...\nspam\tFree entry in 2 a wkly comp\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Ok lar... Joking wif u oni...", 0), ("Free entry in 2 a wkly comp", 1)])

    def test_headerless_csv(self):
        path = self.write("a.csv", "ham,Hello there\nspam,Free prize\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Hello there", 0), ("Free prize", 1)])

    def test_tsv_quote_mark_does_not_swallow_following_rows(self):
        # In a TSV, a message that starts with a quote must stay a single row
        path = self.write("a.tsv", 'ham\t"Hello there\nspam\tFree entry now\nham\tSee you\n')
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0], ('"Hello there', 0))

    def test_quoted_csv_field_containing_a_comma(self):
        path = self.write("a.csv", 'label,text\nspam,"Win cash, now"\n')
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Win cash, now", 1)])

    def test_blank_lines_are_ignored(self):
        path = self.write("a.csv", "label,text\n\nspam,Free prize\n\n   \nham,See you\n")
        data, output = capture(sc.load_dataset, path)
        self.assertEqual(len(data), 2)
        self.assertEqual(output, "")

    def test_whitespace_is_trimmed(self):
        path = self.write("a.csv", "label,text\n  spam  ,  Free prize  \n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Free prize", 1)])

    def test_invalid_rows_are_skipped_with_a_warning(self):
        path = self.write(
            "a.csv",
            "label,text\n"
            "spam,Free prize\n"
            "maybe,Unknown label\n"   # unknown label
            "spam,\n"                 # empty text
            "ham\n"                   # too few columns
            "ham,See you soon\n",
        )
        data, output = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Free prize", 1), ("See you soon", 0)])
        self.assertIn("skipped 3", output)

    def test_utf8_bom_is_handled(self):
        # Excel saves CSV files with a BOM; the header must still be recognised
        path = self.write_bytes("a.csv", b"\xef\xbb\xbflabel,text\nspam,Free prize\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Free prize", 1)])

    def test_invalid_bytes_do_not_crash(self):
        path = self.write_bytes("a.csv", b"label,text\nspam,caf\xe9 offer\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][1], 1)

    def test_unicode_text_is_preserved(self):
        path = self.write("a.csv", "label,text\nspam,Gratis \U0001F381 gewinnen\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [("Gratis \U0001F381 gewinnen", 1)])

    def test_header_only_file_returns_no_rows(self):
        path = self.write("a.csv", "label,text\n")
        data, _ = capture(sc.load_dataset, path)
        self.assertEqual(data, [])

    def test_missing_file_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read"):
            sc.load_dataset(os.path.join(self.tmp_dir, "does_not_exist.csv"))

    def test_directory_instead_of_file_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Could not read"):
            sc.load_dataset(self.tmp_dir)

    def test_empty_file_raises_value_error(self):
        path = self.write("a.csv", "")
        with self.assertRaisesRegex(ValueError, "empty"):
            sc.load_dataset(path)

    def test_file_with_only_blank_lines_raises_value_error(self):
        path = self.write("a.csv", "\n\n   \n")
        with self.assertRaisesRegex(ValueError, "empty"):
            sc.load_dataset(path)

    def test_unrecognised_columns_raise_value_error(self):
        path = self.write("a.csv", "foo,bar\n1,2\n")
        with self.assertRaisesRegex(ValueError, "label and text columns"):
            sc.load_dataset(path)

    def test_label_column_without_text_column_raises_value_error(self):
        path = self.write("a.csv", "label,other\nspam,x\n")
        with self.assertRaisesRegex(ValueError, "label and text columns"):
            sc.load_dataset(path)


class ValidateDatasetTests(unittest.TestCase):
    def test_builtin_data_is_valid(self):
        _, output = capture(sc.validate_dataset, sc.DATA)
        self.assertIn(f"Training on {len(sc.DATA)} messages", output)

    def test_prints_class_counts(self):
        _, output = capture(sc.validate_dataset, make_data(spam=7, ham=9))
        self.assertIn("Training on 16 messages (7 spam, 9 not spam)", output)

    def test_exactly_minimum_per_class_is_accepted(self):
        n = sc.MIN_PER_CLASS
        capture(sc.validate_dataset, make_data(spam=n, ham=n))  # must not raise

    def test_one_below_minimum_is_rejected(self):
        n = sc.MIN_PER_CLASS
        with self.assertRaisesRegex(ValueError, "'spam'"):
            sc.validate_dataset(make_data(spam=n - 1, ham=n))
        with self.assertRaisesRegex(ValueError, "'not spam'"):
            sc.validate_dataset(make_data(spam=n, ham=n - 1))

    def test_single_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "'not spam'"):
            sc.validate_dataset(make_data(spam=20, ham=0))

    def test_empty_data_is_rejected(self):
        with self.assertRaises(ValueError):
            sc.validate_dataset([])


class TrainTests(TempFileTestCase):
    @classmethod
    def setUpClass(cls):
        # Train the built-in model once; training is deterministic so tests can share it
        cls.model, cls.train_output = capture(sc.train)

    def test_reports_test_accuracy(self):
        self.assertIn("Test accuracy:", self.train_output)

    def test_builtin_model_separates_obvious_spam_and_ham(self):
        spam_prob = self.model.predict_proba(["Win a free prize now click here"])[0][1]
        ham_prob = self.model.predict_proba(["Please review my pull request when you get time"])[0][1]
        self.assertGreater(spam_prob, 0.5)
        self.assertLess(ham_prob, 0.5)

    def test_probabilities_are_valid_for_unusual_input(self):
        messages = ["", "   ", "???", "12345", "a" * 10000, "\u00e9\u00e8 \U0001F381", "the and of"]
        for message in messages:
            with self.subTest(message=message[:20]):
                probs = self.model.predict_proba([message])[0]
                self.assertAlmostEqual(probs.sum(), 1.0)
                self.assertTrue(0.0 <= probs[1] <= 1.0)

    def test_training_is_deterministic(self):
        # The final model is retrained on all data, so the random train/test split
        # only shows up in the printed evaluation report. Compare that report and
        # the predictions across several runs.
        message = ["Win a free prize now click here"]
        expected = self.model.predict_proba(message).tolist()
        for _ in range(5):
            model, output = capture(sc.train)
            self.assertEqual(output, self.train_output)
            self.assertEqual(model.predict_proba(message).tolist(), expected)

    def test_invalid_data_raises_value_error(self):
        with self.assertRaises(ValueError):
            sc.train([("only one message", 1)])
        with self.assertRaises(ValueError):
            sc.train([])

    def test_train_on_custom_dataset_end_to_end(self):
        path = self.write("custom.csv", make_csv(spam=10, ham=10))
        data, _ = capture(sc.load_dataset, path)
        model, output = capture(sc.train, data)
        self.assertIn("Training on 20 messages", output)
        self.assertGreater(model.predict_proba(["claim your free cash prize"])[0][1], 0.5)
        self.assertLess(model.predict_proba(["lunch meeting notes for the project"])[0][1], 0.5)

    def test_train_on_headerless_tsv_end_to_end(self):
        lines = [f"{'spam' if label else 'ham'}\t{text}" for text, label in make_data(8, 8)]
        path = self.write("custom.tsv", "\n".join(lines) + "\n")
        data, _ = capture(sc.load_dataset, path)
        model, _ = capture(sc.train, data)
        self.assertGreater(model.predict_proba(["free cash prize winner"])[0][1], 0.5)


class ParseArgsTests(unittest.TestCase):
    def test_data_defaults_to_none(self):
        with mock.patch("sys.argv", ["spam_classifier.py"]):
            self.assertIsNone(sc.parse_args().data)

    def test_data_option_is_parsed(self):
        with mock.patch("sys.argv", ["spam_classifier.py", "--data", "messages.csv"]):
            self.assertEqual(sc.parse_args().data, "messages.csv")

    def test_unknown_option_exits_with_usage_error(self):
        with mock.patch("sys.argv", ["spam_classifier.py", "--bogus"]):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    sc.parse_args()
        self.assertEqual(ctx.exception.code, 2)


class MainTests(TempFileTestCase):
    def run_main(self, inputs, argv=None):
        """Run main() with fake command-line arguments and fake typed input.

        Returns (input_mock, printed_output) so tests can check how many times
        the prompt was shown.
        """
        argv = ["spam_classifier.py"] + (argv or [])
        with mock.patch("sys.argv", argv), mock.patch("builtins.input", side_effect=inputs) as fake_input:
            _, output = capture(sc.main)
        return fake_input, output

    @staticmethod
    def result_lines(output):
        return [line for line in output.splitlines() if "spam probability" in line]

    def test_default_run_classifies_messages(self):
        _, output = self.run_main([
            "Win a free prize now click here",
            "Please review my pull request when you get time",
            "quit",
        ])
        spam_line, ham_line = self.result_lines(output)
        self.assertTrue(spam_line.startswith("SPAM"))
        self.assertTrue(ham_line.startswith("NOT SPAM"))

    def test_exit_words_stop_the_loop(self):
        for word in ("quit", "QUIT", "exit", "  Exit  ", ""):
            with self.subTest(word=word):
                fake_input, _ = self.run_main([word, "this should never be read"])
                self.assertEqual(fake_input.call_count, 1)

    def test_end_of_input_exits_cleanly(self):
        self.run_main(EOFError)  # must not raise

    def test_ctrl_c_exits_cleanly(self):
        self.run_main(KeyboardInterrupt)  # must not raise

    def test_data_option_trains_on_the_given_file(self):
        path = self.write("custom.csv", make_csv(spam=10, ham=10))
        _, output = self.run_main(["claim your free cash prize", "quit"], argv=["--data", path])
        self.assertIn("Training on 20 messages", output)
        self.assertTrue(self.result_lines(output)[0].startswith("SPAM"))

    def test_missing_data_file_exits_with_error_message(self):
        missing = os.path.join(self.tmp_dir, "nope.csv")
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["quit"], argv=["--data", missing])
        self.assertTrue(str(ctx.exception.code).startswith("Error: Could not read"))

    def test_too_few_examples_exits_with_error_message(self):
        path = self.write("tiny.csv", make_csv(spam=2, ham=2))
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(["quit"], argv=["--data", path])
        self.assertTrue(str(ctx.exception.code).startswith("Error: Need at least"))


if __name__ == "__main__":
    unittest.main()
