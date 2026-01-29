"""Tests for legiscope.utils.str2bool."""

import pytest
import argparse
from legiscope.utils import str2bool

class TestStr2Bool:
    """Test str2bool function."""

    @pytest.mark.parametrize("value,expected", [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("t", True),
        ("T", True),
        ("yes", True),
        ("YES", True),
        ("y", True),
        ("Y", True),
        ("1", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("f", False),
        ("F", False),
        ("no", False),
        ("NO", False),
        ("n", False),
        ("N", False),
        ("0", False),
    ])
    def test_valid_inputs(self, value, expected):
        """Test valid boolean string representations."""
        assert str2bool(value) == expected

    @pytest.mark.parametrize("value", [
        "maybe",
        "2",
        "foo",
        "",
        None,
    ])
    def test_invalid_inputs(self, value):
        """Test invalid inputs raise ArgumentTypeError (if user provided) or TypeError."""
        # argparse would catch this in real usage, but str2bool raises ArgumentTypeError
        # or AttributeError if not string/bool.
        if value is None:
             # This depends on implementation if it expects strict string
             # The implementation uses v.lower() so it will raise AttributeError for None
             with pytest.raises(AttributeError):
                 str2bool(value)
        else:
            with pytest.raises(argparse.ArgumentTypeError):
                str2bool(value)
