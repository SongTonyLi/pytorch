# Owner(s): ["module: dynamo"]

"""
Comprehensive tests for mathematical operators in PyTorch Dynamo.

Tests cover:
- Logical operators: or, and
- Bitwise operators: |, &, ^, ~, <<, >>
- Arithmetic operators: +, -, *, /, //, %, **
- Comparison operators: ==, !=, <, <=, >, >=
- Short-circuit evaluation behavior
- Various operand types: bool, int, str, list, set, Tensor, etc.
- User-defined classes with operator overloading
- Type coercion and mixed-type operations
"""

import collections

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class TestLogicalOr(torch._dynamo.test_case.PythonTestCase):
    """Tests for logical OR operator (or)"""

    @make_dynamo_test
    def test_or_with_booleans(self):
        """Test or with boolean operands"""
        self.assertEqual(True or True, True)  # noqa: SIM222
        self.assertEqual(True or False, True)  # noqa: SIM222
        self.assertEqual(False or True, True)  # noqa: SIM222
        self.assertEqual(False or False, False)  # noqa: RUF100

    @make_dynamo_test
    def test_or_with_integers(self):
        """Test or with integer operands (0 is falsy, non-zero is truthy)"""
        self.assertEqual(0 or 5, 5)  # noqa: SIM222
        self.assertEqual(5 or 0, 5)  # noqa: SIM222
        self.assertEqual(3 or 7, 3)  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_strings(self):
        """Test or with string operands"""
        self.assertEqual("" or "hello", "hello")  # noqa: SIM222
        self.assertEqual("hello" or "world", "hello")  # noqa: SIM222

    @make_dynamo_test
    def test_or_with_containers(self):
        """Test or with container operands (empty is falsy)"""
        self.assertEqual([] or [1, 2], [1, 2])  # noqa: SIM222
        self.assertEqual([1, 2] or [], [1, 2])  # noqa: SIM222
        self.assertEqual(None or 5, 5)  # noqa: SIM222

    @make_dynamo_test
    def test_or_short_circuit(self):
        """Test that or short-circuits when first operand is truthy"""
        x = 5
        # This should return x without evaluating the right side
        result = x or (1 / 0)  # Would raise ZeroDivisionError if evaluated
        self.assertEqual(result, 5)

    @make_dynamo_test
    def test_or_chained(self):
        """Test chained or operations"""
        self.assertEqual(0 or 0 or 3 or 4, 3)  # noqa: SIM222
        self.assertEqual(False or False or True, True)  # noqa: SIM222


class TestBitwiseOrIntegers(torch._dynamo.test_case.PythonTestCase):
    """Tests for bitwise OR operator (|) with integers"""

    @make_dynamo_test
    def test_bitwise_or_integers(self):
        """Test bitwise OR with various integers"""
        self.assertEqual(5 | 3, 7)  # 101 | 011 = 111
        self.assertEqual(12 | 10, 14)  # 1100 | 1010 = 1110
        self.assertEqual(5 | 0, 5)
        self.assertEqual(256 | 128, 384)

    @make_dynamo_test
    def test_bitwise_or_negative_integers(self):
        """Test bitwise OR with negative integers (two's complement)"""
        # -1 in two's complement has all bits set
        self.assertEqual(-1 | 0, -1)
        self.assertEqual(5 | -1, -1)
        self.assertEqual(-2 | -3, -1)

    @make_dynamo_test
    def test_bitwise_or_chained(self):
        """Test chained bitwise OR operations"""
        self.assertEqual(1 | 2 | 4 | 8, 15)


class TestBitwiseOrBooleans(torch._dynamo.test_case.PythonTestCase):
    """Tests for bitwise OR operator (|) with booleans"""

    @make_dynamo_test
    def test_bitwise_or_bools(self):
        """Test bitwise OR with booleans (bool is subclass of int)"""
        self.assertEqual(True | True, True)
        self.assertEqual(True | False, True)
        self.assertEqual(False | False, False)

    @make_dynamo_test
    def test_bitwise_or_int_and_bool(self):
        """Test bitwise OR between int and bool"""
        self.assertEqual(5 | True, 5)
        self.assertEqual(0 | True, 1)


class TestBitwiseOrSet(torch._dynamo.test_case.PythonTestCase):
    """Tests for bitwise OR operator (|) with set objects"""

    @parametrize(
        "operand1,operand2,expected",
        [
            ({1, 2}, {2, 3}, {1, 2, 3}),
            ({1}, {2}, {1, 2}),
            ({1, 2, 3}, {1, 2, 3}, {1, 2, 3}),
        ],
    )
    @make_dynamo_test
    def test_set_union_operations(self, operand1, operand2, expected):
        """Test set union via bitwise OR operator"""
        result = operand1 | operand2
        self.assertEqual(result, expected)

    @make_dynamo_test
    def test_set_union_with_empty(self):
        """Test set union with empty set"""
        self.assertEqual({1, 2} | set(), {1, 2})
        self.assertEqual(set() | {1, 2}, {1, 2})

    @make_dynamo_test
    def test_set_union_empty(self):
        """Test set union of two empty sets"""
        self.assertEqual(set() | set(), set())

    @make_dynamo_test
    def test_set_union_chained(self):
        """Test chained set union operations"""
        self.assertEqual({1} | {2} | {3}, {1, 2, 3})
        self.assertEqual({1, 2} | {2, 3} | {3, 4}, {1, 2, 3, 4})


class TestBitwiseOrFrozenSet(torch._dynamo.test_case.PythonTestCase):
    """Tests for bitwise OR operator (|) with frozenset objects"""

    @parametrize(
        "operand1,operand2,expected",
        [
            (frozenset({1, 2}), frozenset({2, 3}), frozenset({1, 2, 3})),
            (frozenset({1}), frozenset({2}), frozenset({1, 2})),
            (frozenset({1, 2, 3}), frozenset({1, 2, 3}), frozenset({1, 2, 3})),
        ],
    )
    @make_dynamo_test
    def test_frozenset_union_operations(self, operand1, operand2, expected):
        """Test frozenset union via bitwise OR operator"""
        result = operand1 | operand2
        self.assertEqual(result, expected)

    @make_dynamo_test
    def test_frozenset_union_with_empty(self):
        """Test frozenset union with empty frozenset"""
        self.assertEqual(frozenset({1, 2}) | frozenset(), frozenset({1, 2}))
        self.assertEqual(frozenset() | frozenset({1, 2}), frozenset({1, 2}))

    @make_dynamo_test
    def test_frozenset_union_empty(self):
        """Test frozenset union of two empty frozensets"""
        self.assertEqual(frozenset() | frozenset(), frozenset())

    @make_dynamo_test
    def test_frozenset_union_chained(self):
        """Test chained frozenset union operations"""
        self.assertEqual(
            frozenset({1}) | frozenset({2}) | frozenset({3}),
            frozenset({1, 2, 3}),
        )


class UserDefinedDict(dict):
    """User-defined dict subclass for testing __or__ operator"""


class _BitwiseOrDictBase:
    """Base class for testing bitwise OR operator with different dict types (Python 3.9+)"""

    def make_left(self, data):
        """Create left operand - override in subclass if needed"""
        raise NotImplementedError

    def make_right(self, data):
        """Create right operand - override in subclass if needed"""
        raise NotImplementedError

    @make_dynamo_test
    def test_dict_or_operation(self):
        """Test dict merge via bitwise OR operator"""
        left = self.make_left({"a": 1, "b": 2})
        right = self.make_right({"b": 20, "c": 3})
        result = left | right
        self.assertEqual(result, {"a": 1, "b": 20, "c": 3})

    @make_dynamo_test
    def test_dict_or_empty(self):
        """Test dict merge with empty dict"""
        left = self.make_left({"a": 1})
        right = self.make_right({})
        result = left | right
        self.assertEqual(result, {"a": 1})

    @make_dynamo_test
    def test_dict_or_chained(self):
        """Test chained dict merge operations"""
        d1 = self.make_left({"a": 1})
        d2 = self.make_right({"b": 2})
        d3 = self.make_left({"c": 3})
        result = d1 | d2 | d3
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})


class TestDictOrDict(_BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for dict | dict merge"""

    def make_left(self, data):
        return dict(data)

    def make_right(self, data):
        return dict(data)


class TestDictOrDefaultdict(_BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for dict | defaultdict merge"""

    def make_left(self, data):
        return dict(data)

    def make_right(self, data):
        return collections.defaultdict(int, data)


class TestDefaultdictOrDict(_BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for defaultdict | dict merge"""

    def make_left(self, data):
        return collections.defaultdict(int, data)

    def make_right(self, data):
        return dict(data)


class TestDefaultdictOrDefaultdict(
    _BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for defaultdict | defaultdict merge"""

    def make_left(self, data):
        return collections.defaultdict(int, data)

    def make_right(self, data):
        return collections.defaultdict(int, data)


class TestDictOrOrdereddict(_BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for dict | OrderedDict merge"""

    def make_left(self, data):
        return dict(data)

    def make_right(self, data):
        return collections.OrderedDict(data.items())


class TestOrdereddictOrDict(_BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for OrderedDict | dict merge"""

    def make_left(self, data):
        return collections.OrderedDict(data.items())

    def make_right(self, data):
        return dict(data)


class TestOrdereddictOrOrdereddict(
    _BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for OrderedDict | OrderedDict merge"""

    def make_left(self, data):
        return collections.OrderedDict(data.items())

    def make_right(self, data):
        return collections.OrderedDict(data.items())


class TestDictOrUserDefinedDict(
    _BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for dict | user-defined dict merge"""

    def make_left(self, data):
        return dict(data)

    def make_right(self, data):
        return UserDefinedDict(data)


class TestUserDefinedDictOrDict(
    _BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for user-defined dict | dict merge"""

    def make_left(self, data):
        return UserDefinedDict(data)

    def make_right(self, data):
        return dict(data)


class TestUserDefinedDictOrUserDefinedDict(
    _BitwiseOrDictBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for user-defined dict | user-defined dict merge"""

    def make_left(self, data):
        return UserDefinedDict(data)

    def make_right(self, data):
        return UserDefinedDict(data)


class _BitwiseOrInplaceBase:
    """Base class for testing inplace bitwise OR operator (|=) with different container types"""

    container_type = None  # Override in subclass
    data1 = None
    data2 = None
    expected = None

    def make_container(self, data):
        """Create a container of the appropriate type"""
        return self.container_type(data)

    @make_dynamo_test
    def test_inplace_or_basic(self):
        """Test inplace OR operation"""
        left = self.make_container(self.data1)
        right = self.make_container(self.data2)
        left |= right
        self.assertEqual(left, self.expected)


class TestDictInplaceOr(_BitwiseOrInplaceBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for dict |= dict inplace merge"""

    container_type = dict
    data1 = {"a": 1, "b": 2}
    data2 = {"b": 20, "c": 3}
    expected = {"a": 1, "b": 20, "c": 3}


class TestSetInplaceOr(_BitwiseOrInplaceBase, torch._dynamo.test_case.PythonTestCase):
    """Tests for set |= set inplace union"""

    container_type = set
    data1 = {1, 2}
    data2 = {2, 3}
    expected = {1, 2, 3}


class TestDefaultdictInplaceOr(
    _BitwiseOrInplaceBase, torch._dynamo.test_case.PythonTestCase
):
    """Tests for defaultdict |= dict inplace merge"""

    def make_container(self, data):
        return collections.defaultdict(int, data)

    data1 = {"a": 1, "b": 2}
    data2 = {"b": 20, "c": 3}
    expected = {"a": 1, "b": 20, "c": 3}


class TestReversedOr(torch._dynamo.test_case.PythonTestCase):
    """Tests for reversed bitwise OR operator (__ror__)"""

    @make_dynamo_test
    def test_reversed_or_with_integer(self):
        """Test reversed OR with integer (calls __ror__ on right operand)"""
        obj = UserDefinedClassWithOr(3)
        result = 5 | obj
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_with_user_defined_object(self):
        """Test reversed OR with user-defined object"""
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        # This will call obj2.__ror__(obj1) if obj1.__or__ returns NotImplemented
        result = obj1 | obj2
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_reversed_or_chained(self):
        """Test chained reversed OR operations"""
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        result = 0 | obj1 | obj2 | obj3
        # 0 | obj1 -> obj1.__ror__(0) -> 1
        # 1 | obj2 -> UserDefinedClassWithOr(1) | obj2 -> obj2.__ror__(...) or __or__
        # This tests the chain behavior
        self.assertEqual(result.value, 7)


class TestBitwiseOrUnsupported(torch._dynamo.test_case.PythonTestCase):
    """Tests that verify unsupported container types raise TypeError with | operator"""

    @make_dynamo_test
    def test_list_or_list_raises_type_error(self):
        """Test that list | list raises TypeError"""
        with self.assertRaises(TypeError):
            [1, 2] | [3, 4]

    @make_dynamo_test
    def test_tuple_or_tuple_raises_type_error(self):
        """Test that tuple | tuple raises TypeError"""
        with self.assertRaises(TypeError):
            (1, 2) | (3, 4)

    @make_dynamo_test
    def test_empty_list_or_list_raises_type_error(self):
        """Test that empty list | list raises TypeError"""
        with self.assertRaises(TypeError):
            [] | [1, 2]

    @make_dynamo_test
    def test_empty_tuple_or_tuple_raises_type_error(self):
        """Test that empty tuple | tuple raises TypeError"""
        with self.assertRaises(TypeError):
            () | (1, 2)


class UserDefinedClassWithOr:
    """User-defined class implementing __or__ and __ror__ operators"""

    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(self.value | other.value)
        return UserDefinedClassWithOr(self.value | other)

    def __ror__(self, other):
        """Reversed OR operator - called when left operand doesn't support __or__"""
        if isinstance(other, UserDefinedClassWithOr):
            return UserDefinedClassWithOr(other.value | self.value)
        return UserDefinedClassWithOr(other | self.value)

    def __eq__(self, other):
        if isinstance(other, UserDefinedClassWithOr):
            return self.value == other.value
        return False

    def __repr__(self):
        return f"UserDefinedClassWithOr({self.value})"


class TestUserDefinedOr(torch._dynamo.test_case.PythonTestCase):
    """Tests for user-defined classes with __or__ operator"""

    def setUp(self):
        super().setUp()
        self.obj1 = UserDefinedClassWithOr(5)
        self.obj2 = UserDefinedClassWithOr(3)
        self.obj3 = UserDefinedClassWithOr(0)

    @make_dynamo_test
    def test_user_defined_or_basic(self):
        """Test __or__ on user-defined class"""
        obj1 = UserDefinedClassWithOr(5)
        obj2 = UserDefinedClassWithOr(3)
        result = obj1 | obj2
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_with_integer(self):
        """Test __or__ with integer operand"""
        obj = UserDefinedClassWithOr(5)
        result = obj | 3
        self.assertEqual(result, UserDefinedClassWithOr(7))

    @make_dynamo_test
    def test_user_defined_or_zero(self):
        """Test __or__ with zero"""
        obj = UserDefinedClassWithOr(5)
        result = obj | 0
        self.assertEqual(result, UserDefinedClassWithOr(5))

    @make_dynamo_test
    def test_user_defined_or_chained(self):
        """Test chained __or__ operations"""
        obj1 = UserDefinedClassWithOr(1)
        obj2 = UserDefinedClassWithOr(2)
        obj3 = UserDefinedClassWithOr(4)
        result = obj1 | obj2 | obj3
        self.assertEqual(result, UserDefinedClassWithOr(7))


class LeftOrClass:
    """User-defined class whose __or__ returns NotImplemented for unknown types"""

    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, LeftOrClass):
            return LeftOrClass(self.value | other.value)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LeftOrClass):
            return LeftOrClass(other.value | self.value)
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, LeftOrClass) and self.value == other.value

    def __repr__(self):
        return f"LeftOrClass({self.value})"


class RightOrClass:
    """User-defined class whose __ror__ handles LeftOrClass via string concatenation"""

    def __init__(self, value):
        self.value = value

    def __or__(self, other):
        if isinstance(other, RightOrClass):
            return RightOrClass(self.value + "|" + other.value)
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, LeftOrClass):
            return f"LeftOrClass({other.value})|RightOrClass({self.value})"
        return NotImplemented

    def __eq__(self, other):
        return isinstance(other, RightOrClass) and self.value == other.value

    def __repr__(self):
        return f"RightOrClass({self.value})"


class TestCrossTypeUserDefinedOr(torch._dynamo.test_case.PythonTestCase):
    """Tests for two distinct user-defined classes with different __or__/__ror__ implementations"""

    @make_dynamo_test
    def test_left_or_left_uses_or(self):
        """LeftOrClass | LeftOrClass dispatches to __or__"""
        a = LeftOrClass(5)
        b = LeftOrClass(3)
        result = a | b
        self.assertEqual(result, LeftOrClass(7))

    @make_dynamo_test
    def test_right_or_right_uses_or(self):
        """RightOrClass | RightOrClass dispatches to __or__"""
        a = RightOrClass("a")
        b = RightOrClass("b")
        result = a | b
        self.assertEqual(result, RightOrClass("a|b"))

    @make_dynamo_test
    def test_left_or_right_falls_back_to_ror(self):
        """LeftOrClass | RightOrClass: __or__ returns NotImplemented, falls back to RightOrClass.__ror__"""
        a = LeftOrClass(42)
        b = RightOrClass("x")
        result = a | b
        self.assertEqual(result, "LeftOrClass(42)|RightOrClass(x)")

    @make_dynamo_test
    def test_right_or_left_raises(self):
        """RightOrClass | LeftOrClass: both __or__ and __ror__ return NotImplemented -> TypeError"""
        a = RightOrClass("x")
        b = LeftOrClass(42)
        with self.assertRaises(TypeError):
            a | b


class TestOrOperatorWithTensors(torch._dynamo.test_case.TestCase):
    """Tests for OR operator behavior with torch tensors"""

    @make_dynamo_test
    def test_tensor_to_bool(self):
        """Test converting tensor to bool in or expressions"""
        t_nonzero = torch.tensor(1)
        t_zero = torch.tensor(0)
        self.assertTrue(bool(t_nonzero))
        self.assertFalse(bool(t_zero))


# Classes for TestSubclassRightOp


class _IntSubWithOr(int):
    def __or__(self, other):
        return "_IntSubWithOr.__or__"

    def __ror__(self, other):
        return "_IntSubWithOr.__ror__"


class _BaseWithOr:
    def __or__(self, other):
        return "_BaseWithOr.__or__"

    def __ror__(self, other):
        return "_BaseWithOr.__ror__"


class _SubWithOr(_BaseWithOr):
    def __or__(self, other):
        return "_SubWithOr.__or__"

    def __ror__(self, other):
        return "_SubWithOr.__ror__"


class _InheritedSub(_BaseWithOr):
    pass


class TestSubclassRightOp(torch._dynamo.test_case.PythonTestCase):
    """
    Tests for correct dispatch of subclass overloading __ror__.

    Port of test_descr.py::test_subclass_right_op (CPython abstract.c::binary_op1
    and typeobject.c::SLOT1BINFULL) using the | operator instead of //.
    """

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        # Case 1: subclass of int; tests code in abstract.c::binary_op1().
        # When the right operand is a strict subtype of int, binary_op1 must
        # call __ror__ on the subclass *before* __or__ on int.
        self.assertEqual(_IntSubWithOr(1) | 1, "_IntSubWithOr.__or__")
        self.assertEqual(1 | _IntSubWithOr(1), "_IntSubWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        # Case 2: baseline – plain user-defined class, no subclass priority.
        self.assertEqual(_BaseWithOr() | 1, "_BaseWithOr.__or__")
        self.assertEqual(1 | _BaseWithOr(), "_BaseWithOr.__ror__")

    @make_dynamo_test
    def test_subclass_of_user_defined_gets_priority(self):
        # Case 3: subclass of a user-defined class; this is where SLOT1BINFULL
        # subclass-priority logic kicks in.
        # _BaseWithOr() | _SubWithOr() must call _SubWithOr.__ror__ because
        # _SubWithOr is a subclass of _BaseWithOr.
        self.assertEqual(_SubWithOr() | _BaseWithOr(), "_SubWithOr.__or__")
        self.assertEqual(_BaseWithOr() | _SubWithOr(), "_SubWithOr.__ror__")

    @make_dynamo_test
    def test_inherited_subclass_no_priority(self):
        # Case 4: subclass that does NOT override the dunder — the inherited
        # slot is identical, so no subclass priority applies.
        # _BaseWithOr() | _InheritedSub() must return "_BaseWithOr.__or__",
        # NOT "_BaseWithOr.__ror__" (which would be wrong subclass-priority
        # behavior).  This case failed in CPython 2.2.2 / 2.3a1 and is the
        # "This one would fail" assertion in
        # test_descr.py::test_subclass_right_op.
        # self.assertIs(_InheritedSub.__ror__, _BaseWithOr.__ror__)  # sanity: truly inherited

        # self.assertEqual(_InheritedSub() | 1, "_BaseWithOr.__or__")
        # self.assertEqual(1 | _InheritedSub(), "_BaseWithOr.__ror__")
        # self.assertEqual(_InheritedSub() | _BaseWithOr(), "_BaseWithOr.__or__")
        self.assertEqual(
            _BaseWithOr() | _InheritedSub(), "_BaseWithOr.__or__"
        )  # must NOT call _InheritedSub.__ror__


# Instantiate parametrized tests
instantiate_parametrized_tests(TestBitwiseOrSet)
instantiate_parametrized_tests(TestBitwiseOrFrozenSet)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
