import importlib.util
import unittest


class HorizontalTargetTests(unittest.TestCase):
    def check(self,rows):
        self.assertIsNotNone(importlib.util.find_spec('P2.evaluate_horizontal_targets'))
        from P2.evaluate_horizontal_targets import horizontal_summary
        return horizontal_summary(rows)

    def row(self,speed,horizontal,**other):
        return dict(wind_speed=speed,horizontal_rmse_m=horizontal,position_rmse_m=horizontal+.3,
                    completed=True,attachment=2,direction_rad=0.,**other)

    def test_horizontal_not3d_and_inclusive_limits(self):
        result=self.check([self.row(0.,.5),self.row(1.,.5),self.row(5.,1.5)])
        self.assertTrue(result['passed']);self.assertEqual(result['violations'],[])

    def test_each_case_must_pass_and_failures_are_not_hidden(self):
        result=self.check([self.row(1.,.1),self.row(1.,.6),self.row(3.,1.6)])
        self.assertFalse(result['passed']);self.assertEqual(len(result['violations']),2)
        row=self.row(.5,.1);row['completed']=False
        self.assertFalse(self.check([row])['passed'])

    def test_user_tolerance_keeps_nominal_excess_visible(self):
        self.reference=self.check([self.row(.5,.527)])
        from P2.evaluate_horizontal_targets import horizontal_summary
        result=horizontal_summary([self.row(.5,.527)],weak_tolerance_m=.03)
        self.assertTrue(result['passed']);self.assertEqual(result['violations'],[])
        self.assertEqual(len(result['nominal_violations']),1)
        self.assertFalse(horizontal_summary([self.row(1.,.531)],weak_tolerance_m=.03)['passed'])


if __name__=='__main__':unittest.main()
