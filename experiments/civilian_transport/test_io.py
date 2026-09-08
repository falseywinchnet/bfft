import csv
import json
from pathlib import Path
import tempfile
import unittest
from .track import track


class PositionOnlyIOTests(unittest.TestCase):
    def write(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'x', 'y', 'z', 'unavailable_true_velocity'])
            writer.writerows(rows)

    def test_causal_prefix_and_ignored_extra_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [[i*.1, i*.2, .03*i*i, 3., 99999] for i in range(7)]
            self.write(root/'a.csv', rows)
            changed = [r[:] for r in rows]
            for r in changed:
                r[-1] = -54321
            changed[-1][1] = 1000.
            self.write(root/'b.csv', changed)
            for name in ('a', 'b'):
                track(root/(name+'.csv'), root/(name+'.json'), 'geometric',
                      {'sigma': .35}, particles=64, horizon=.2, query_dt=.1)
            a, b = [json.loads((root/(name+'.json')).read_text())['estimates'] for name in ('a', 'b')]
            self.assertEqual(a[:-1], b[:-1])
            self.assertNotEqual(a[-1], b[-1])

    def test_missing_position_and_invalid_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write(root/'data.csv', [[0., 0., 0., 3., 0.], [.1, '', '', '', 0.], [.2, .2, .1, 3., 0.]])
            track(root/'data.csv', root/'out.json', 'cv', {'sigma': .35}, horizon=.1)
            self.assertEqual(len(json.loads((root/'out.json').read_text())['estimates']), 3)
            self.write(root/'data.csv', [[0., 0., 0., 3., 0.], [0., 1., 1., 3., 0.]])
            with self.assertRaises(ValueError):
                track(root/'data.csv', root/'out.json', 'cv', {'sigma': .35})


if __name__ == '__main__':
    unittest.main()
