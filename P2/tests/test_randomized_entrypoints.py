import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from P2 import evaluate_flexible_rope
from P2.hover_bptt.io import read, write

ROOT = Path(__file__).resolve().parents[1]


class RandomizedEntrypointTests(unittest.TestCase):
    def test_development_output_contains_rope_metadata_for_the_replay_entrypoint(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as folder:
            folder=Path(folder)
            protocol=read(ROOT/'domain_randomization_protocol.json')
            protocol['domain_randomization']['sample_count']=2
            protocol['development_directions_deg']=[45]
            protocol['development_winds']=[[1.,'steady']]
            path=folder/'protocol.json'
            write(path,protocol)
            argv=['evaluate_flexible_rope','--protocol',str(path),
                '--checkpoint',str(ROOT/protocol['parent_checkpoint']),
                '--output',str(folder/'comparison'),'--seconds','.1','--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):
                evaluate_flexible_rope.main()
            for domain in ('nominal','random_1'):
                result=read(folder/'comparison'/domain/'after/result.json')
                self.assertIn('rope',result)
                self.assertEqual(result['rope']['segments'],12)
                self.assertEqual(result['duration_s'],.1)


if __name__=='__main__':
    unittest.main()
