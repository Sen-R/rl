import pytest
from rl.stochasticprocesses import *

class TestOUProcess:
    def test_construction(self):
        self.ou = OUProcess(x_inf=1., time_const=6, std_dev=0.2)

    @pytest.mark.todo
    def test_customize_start(self):
        pass

mu = np.array([1., 2.])
sigma = np.array(0.2)
    
class TestGWNProcess:
    def test_output_shape(self):
        gwn = GaussianWhiteNoiseProcess(mu=mu, sigma=sigma)
        x = gwn.sample()
        assert x.shape == mu.shape
        
    @pytest.mark.todo
    def test_seed_setting(self):
        pass

    @pytest.mark.todo
    def test_univariate_moments(self):
        pass

    @pytest.mark.todo
    def test_bivariate_moments(self):
        pass
