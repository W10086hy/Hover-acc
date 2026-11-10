import os,tempfile,unittest
from pathlib import Path
from px4_offboard.deploy_vm import clear_existing

class DeploymentTests(unittest.TestCase):
    @unittest.skipIf(os.name=='nt','Guest Linux symlink semantics')
    def test_cleanup_unlinks_task_entries_without_following_outside_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);outside=base/'outside';outside.mkdir();(outside/'preserve.txt').write_text('preserve')
            root=base/'task';root.mkdir();sub=root/'old_result';sub.mkdir();(sub/'etc').symlink_to(outside,target_is_directory=True)
            (root/'top_link').symlink_to(outside,target_is_directory=True);(sub/'trace.txt').write_text('old')
            clear_existing(root)
            self.assertEqual(list(root.iterdir()),[])
            self.assertEqual((outside/'preserve.txt').read_text(),'preserve')

if __name__=='__main__':unittest.main()
