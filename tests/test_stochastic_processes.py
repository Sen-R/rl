import pytest
from rl.stochasticprocesses import *

class TestOUProcess:
    def test_construction(self):
        self.ou = OUProcess(x_inf=1., time_const=6, std_dev=0.2)

    @pytest.mark.todo
    def test_customize_start(self):
        pass
