""" Tests of command line program

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from wc_sim import __main__
import abduct
import mock
import unittest
import wc_sim


class CliTestCase(unittest.TestCase):

    def test_raw_cli(self):
        with mock.patch('sys.argv', ['wc-sim', '--help']):
            with self.assertRaises(SystemExit) as context:
                __main__.main()
                self.assertRegex(context.Exception, 'usage: wc-sim')

        with mock.patch('sys.argv', ['wc-sim']):
            with abduct.captured(abduct.out(), abduct.err()) as (stdout, stderr):
                __main__.main()
                self.assertRegex(stdout.getvalue().strip(), 'usage: wc-sim')
                self.assertEqual(stderr.getvalue(), '')

    def test_get_version(self):
        with abduct.captured(abduct.out(), abduct.err()) as (stdout, stderr):
            with __main__.App(argv=['-v']) as app:
                with self.assertRaises(SystemExit):
                    app.run()
            self.assertEqual(stdout.getvalue().strip(), wc_sim.__version__)
            self.assertEqual(stderr.getvalue(), '')

        with abduct.captured(abduct.out(), abduct.err()) as (stdout, stderr):
            with __main__.App(argv=['--version']) as app:
                with self.assertRaises(SystemExit):
                    app.run()
            self.assertEqual(stdout.getvalue().strip(), wc_sim.__version__)
            self.assertEqual(stderr.getvalue(), '')

    def test_migration_handlers(self):
        commands = ['migrate-data', 'do-configured-migration', 'make-data-schema-migration-config-file']
        for command in commands:
            with CaptureOutput(relay=False) as captured:
                with __main__.App(argv=[command, '--help']) as app:
                    with self.assertRaises(SystemExit):
                        app.run()
                self.assertIn('usage: wc-sim {}'.format(command), captured.get_text())
