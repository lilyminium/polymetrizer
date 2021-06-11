
# import pytest
# import numpy as np

# from numpy.testing import assert_array_equal

# from simtk import unit

# import polymetrizer as pet

# UNIT_CHARGE = 1 * unit.elementary_charge


# # @pytest.mark.parametrize("arg, kwargs, result", [
# #     ([1, 2], {}, 1.5),
# #     (np.ones(3) * unit.elementary_charge, {}, UNIT_CHARGE),
# #     ([[UNIT_CHARGE], [UNIT_CHARGE]], {}, UNIT_CHARGE),
# #     ([[UNIT_CHARGE], [UNIT_CHARGE]], {"axis": 0}, [UNIT_CHARGE]),
# #     (np.ones((3, 1)), {"axis": 0}, [1]),
# #     (np.ones((3, 1)) * unit.elementary_charge, {"axis": 0}, [UNIT_CHARGE]),
# #     (np.ones((1, 3)), {"axis": 0}, [1, 1, 1]),
# #     (np.ones((1, 3)) * unit.elementary_charge, {"axis": 0}, [UNIT_CHARGE, UNIT_CHARGE, UNIT_CHARGE]),
# # ])
# # def test_operate_on_quantities(arg, kwargs, result):
# #     print(arg, np.mean(arg),)
# #     calc = pet.ommutils.operate_on_quantities(np.mean, arg, **kwargs)
# #     # assert_array_equal(calc, result)
# #     try:
# #         assert list(calc) == list(result)
# #     except TypeError:
# #         assert calc == result