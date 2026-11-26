"""Submission patches must preserve CRLF bytes through generation and replay."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from bench import patch


class PatchCRLFRoundtripTests(unittest.TestCase):
    def test_crlf_baseline_replays_exact_candidate_bytes(self):
        baseline = b"# context\r\nvalue <- 1\r\n# end\r\n"
        candidates = (
            b"# context\r\nvalue <- 2\r\n# end\r\n",
            b"# context\nvalue <- 2\n# end\n",
        )
        identity = dict(os.environ, GIT_AUTHOR_NAME="test", GIT_AUTHOR_EMAIL="test@local",
                        GIT_COMMITTER_NAME="test", GIT_COMMITTER_EMAIL="test@local")
        for candidate in candidates:
            with self.subTest(candidate_crlf=b"\r\n" in candidate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source"
                source.mkdir()

                def git(*args):
                    return subprocess.run(["git", *args], cwd=source, env=identity,
                                          capture_output=True, check=True).stdout

                git("init", "-q")
                git("config", "core.autocrlf", "false")
                (source / "functions.R").write_bytes(baseline)
                git("add", ".")
                git("commit", "-qm", "CRLF baseline")
                base = git("rev-parse", "HEAD").decode().strip()
                (source / "functions.R").write_bytes(candidate)
                new_bytes = b"extra <- function(x) x + 1\r\n"
                (source / "new.R").write_bytes(new_bytes)

                submission = root / "submission.patch"
                submission.write_bytes(patch.make_patch(source, base))
                replay = root / "replay"
                patch.clean_clone(source, replay, base)
                patch.apply_patch(replay, submission)

                self.assertEqual((replay / "functions.R").read_bytes(), candidate)
                self.assertEqual((replay / "new.R").read_bytes(), new_bytes)


if __name__ == "__main__":
    unittest.main()
