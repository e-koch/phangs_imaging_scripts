from .test_pipelineLogger import TestingPipelineLogger
from .test_pipelineLogger import TestingPipelineLoggerInCasa
from .test_casaCubeRoutines import TestingCasaCubeRoutines
from .test_casaCubeRoutines import TestingCasaCubeRoutinesInCasa
from .test_casaFeatherRoutines import TestingCasaFeatherRoutines
from .test_casaFeatherRoutines import TestingCasaFeatherRoutinesInCasa
from .test_handlerKeys import TestingHandlerKeys
from .test_handlerKeys import TestingHandlerKeysInCasa
from .test_handlerVis import TestingHandlerVis
from .test_handlerVis import TestingHandlerVisInCasa
from .test_handlerImaging import TestingHandlerImaging
from .test_handlerImaging import TestingHandlerImagingInCasa
from .test_noise_convergence import TestingNoiseConvergence
from .test_noise_convergence import TestingNoiseConvergenceInCasa
from .test_check_noise_convergence import TestCheckNoiseConvergence
from .test_check_noise_convergence import TestingCheckNoiseConvergenceInCasa

import phangsPipeline
import unittest

class TestingAllInCasa():
    """docstring for TestingHandlerKeysInCasa

    To run all tests, run following command in CASA:
        sys.path.append('../analysis_scripts')
        sys.path.append('.')
        import phangsPipelineTests
        phangsPipelineTests.TestingAllInCasa().run()

    """

    def __init__(self):
        pass

    def suite(self=None):
        testsuite = unittest.TestSuite()
        testsuite.addTest(unittest.makeSuite(TestingPipelineLogger))
        testsuite.addTest(unittest.makeSuite(TestingCasaCubeRoutines))
        testsuite.addTest(unittest.makeSuite(TestingCasaFeatherRoutines))
        testsuite.addTest(unittest.makeSuite(TestingHandlerKeys))
        testsuite.addTest(unittest.makeSuite(TestingHandlerVis))
        testsuite.addTest(unittest.makeSuite(TestingHandlerImaging))
        testsuite.addTest(unittest.makeSuite(TestCheckNoiseConvergence))
        testsuite.addTest(unittest.makeSuite(TestingNoiseConvergence))
        return testsuite

    def run(self):
        unittest.main(defaultTest='phangsPipelineTests.TestingAllInCasa.suite',
                      verbosity=2,
                      exit=False)

