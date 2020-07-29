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

@pytest.mark.todo # Need to strengthen this test
@pytest.mark.parametrize('scrambler', [OUScrambler(1000, 4, 6, 0.2),
                                       GaussianWhiteNoiseScrambler(1000, 4,
                                                                   0.2)])
def test_scramblers(scrambler):
        scrambled = scrambler(np.random.rand(1000, 4))
        assert scrambled.shape == (1000, 4)
        assert np.all(scrambled <= 1.)
        assert np.all(scrambled >= -1.)
