from distant_sunburn.json_utils import flatten_json_to_pathmap
import jsonpatch


def test_complex_manual_nested_structure():
    """Test with a manually constructed complex nested JSON structure."""
    # Manually construct a complex nested structure
    data = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "profile": {
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                    "coordinates": [40.7128, -74.0060],
                },
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                    "languages": ["en", "es", "fr"],
                },
            },
        },
        "orders": [
            {
                "id": "order-1",
                "items": [
                    {"name": "laptop", "price": 999.99},
                    {"name": "mouse", "price": 29.99},
                ],
                "status": "shipped",
            },
            {"id": "order-2", "items": [], "status": "pending"},
        ],
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "version": 1.0,
            "active": True,
        },
    }

    result = flatten_json_to_pathmap(data)

    # Manually assert the expected flattened structure
    expected = {
        "/user/id": 123,
        "/user/name": "John Doe",
        "/user/profile/age": 30,
        "/user/profile/address/street": "123 Main St",
        "/user/profile/address/city": "New York",
        "/user/profile/address/coordinates/0": 40.7128,
        "/user/profile/address/coordinates/1": -74.0060,
        "/user/profile/preferences/theme": "dark",
        "/user/profile/preferences/notifications": True,
        "/user/profile/preferences/languages/0": "en",
        "/user/profile/preferences/languages/1": "es",
        "/user/profile/preferences/languages/2": "fr",
        "/orders/0/id": "order-1",
        "/orders/0/items/0/name": "laptop",
        "/orders/0/items/0/price": 999.99,
        "/orders/0/items/1/name": "mouse",
        "/orders/0/items/1/price": 29.99,
        "/orders/0/status": "shipped",
        "/orders/1/id": "order-2",
        "/orders/1/items": [],
        "/orders/1/status": "pending",
        "/metadata/created_at": "2024-01-01T00:00:00Z",
        "/metadata/version": 1.0,
        "/metadata/active": True,
    }

    assert result == expected


def test_empty_structures():
    """Test edge cases with empty dictionaries and lists."""
    # Test with empty dict - gets recorded as terminal value with empty path
    data1 = {}
    result1 = flatten_json_to_pathmap(data1)
    assert result1 == {"": {}}

    # Test with empty list - gets recorded as terminal value with empty path
    data2 = []
    result2 = flatten_json_to_pathmap(data2)
    assert result2 == {"": []}

    # Test with dict containing empty dict
    data3 = {"empty_dict": {}}
    result3 = flatten_json_to_pathmap(data3)
    assert result3 == {"/empty_dict": {}}

    # Test with dict containing empty list
    data4 = {"empty_list": []}
    result4 = flatten_json_to_pathmap(data4)
    assert result4 == {"/empty_list": []}

    # Test with list containing empty dict
    data5 = [{}]
    result5 = flatten_json_to_pathmap(data5)
    assert result5 == {"/0": {}}

    # Test with list containing empty list
    data6 = [[]]
    result6 = flatten_json_to_pathmap(data6)
    assert result6 == {"/0": []}


def test_primitive_values():
    """Test with primitive values (strings, numbers, booleans, null)."""
    data = {
        "string": "hello world",
        "integer": 42,
        "float": 3.14159,
        "boolean_true": True,
        "boolean_false": False,
        "null_value": None,
    }

    result = flatten_json_to_pathmap(data)

    expected = {
        "/string": "hello world",
        "/integer": 42,
        "/float": 3.14159,
        "/boolean_true": True,
        "/boolean_false": False,
        "/null_value": None,
    }

    assert result == expected


def test_mixed_types_and_edge_cases():
    """Test with mixed data types and various edge cases."""
    data = {
        "nested": {
            "list_with_mixed_types": [1, "string", True, None, {"nested": "value"}],
            "dict_with_numeric_keys": {"0": "zero", "1": "one", "2": "two"},
        },
        "list_of_dicts": [{"a": 1, "b": 2}, {"c": 3, "d": 4}],
        "special_chars": {
            "key/with/slashes": "value",
            "key.with.dots": "another_value",
        },
    }

    result = flatten_json_to_pathmap(data)

    expected = {
        "/nested/list_with_mixed_types/0": 1,
        "/nested/list_with_mixed_types/1": "string",
        "/nested/list_with_mixed_types/2": True,
        "/nested/list_with_mixed_types/3": None,
        "/nested/list_with_mixed_types/4/nested": "value",
        "/nested/dict_with_numeric_keys/0": "zero",
        "/nested/dict_with_numeric_keys/1": "one",
        "/nested/dict_with_numeric_keys/2": "two",
        "/list_of_dicts/0/a": 1,
        "/list_of_dicts/0/b": 2,
        "/list_of_dicts/1/c": 3,
        "/list_of_dicts/1/d": 4,
        "/special_chars/key/with/slashes": "value",
        "/special_chars/key.with.dots": "another_value",
    }

    assert result == expected


def test_single_level_structures():
    """Test with single-level structures (no nesting)."""
    # Single dict
    data1 = {"a": 1, "b": 2, "c": 3}
    result1 = flatten_json_to_pathmap(data1)
    expected1 = {"/a": 1, "/b": 2, "/c": 3}
    assert result1 == expected1

    # Single list
    data2 = ["x", "y", "z"]
    result2 = flatten_json_to_pathmap(data2)
    expected2 = {"/0": "x", "/1": "y", "/2": "z"}
    assert result2 == expected2


def test_edit_distance_normalization():
    """Test that edit distance normalization works correctly with JsonPatch."""
    # Create a complex JSON structure with known number of attributes
    source_json = {
        "user": {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",
            "profile": {
                "age": 30,
                "city": "New York",
                "country": "USA",
                "preferences": {"theme": "dark", "language": "en"},
            },
        },
        "settings": {"notifications": True, "privacy": "public"},
    }

    # Create a modified version with 3 specific changes
    destination_json = {
        "user": {
            "id": 456,  # Changed: 123 -> 456
            "name": "John Doe",
            "email": "john.doe@example.com",  # Changed: john@example.com -> john.doe@example.com
            "profile": {
                "age": 30,
                "city": "San Francisco",  # Changed: New York -> San Francisco
                "country": "USA",
                "preferences": {"theme": "dark", "language": "en"},
            },
        },
        "settings": {"notifications": True, "privacy": "public"},
    }

    # Flatten both JSON structures to get all attributes
    source_flat = flatten_json_to_pathmap(source_json)
    destination_flat = flatten_json_to_pathmap(destination_json)

    # Calculate total number of attributes (should be the same for both)
    total_attributes = len(source_flat)
    assert total_attributes == len(
        destination_flat
    ), "Both JSONs should have same number of attributes"

    # Create JsonPatch to find the differences
    patch = jsonpatch.make_patch(source_json, destination_json)
    num_changes = len(patch.patch)

    # Calculate normalized edit distance
    normalized_distance = num_changes / total_attributes

    # Verify our expectations
    # We made 3 changes: id, email, and city
    expected_changes = 3
    expected_normalized_distance = expected_changes / total_attributes

    assert (
        num_changes == expected_changes
    ), f"Expected {expected_changes} changes, got {num_changes}"
    assert (
        normalized_distance == expected_normalized_distance
    ), f"Expected normalized distance {expected_normalized_distance}, got {normalized_distance}"

    # Verify the total number of attributes is what we expect
    # Manual count: /user/id, /user/name, /user/email, /user/profile/age, /user/profile/city,
    # /user/profile/country, /user/profile/preferences/theme, /user/profile/preferences/language,
    # /settings/notifications, /settings/privacy = 10 attributes
    expected_total_attributes = 10
    assert (
        total_attributes == expected_total_attributes
    ), f"Expected {expected_total_attributes} total attributes, got {total_attributes}"

    # Verify the normalized distance is 30% (3 changes out of 10 attributes)
    expected_percentage = 0.3
    assert (
        abs(normalized_distance - expected_percentage) < 0.001
    ), f"Expected ~30% normalized distance, got {normalized_distance:.1%}"

    # Print the patch for verification
    print(f"Total attributes: {total_attributes}")
    print(f"Number of changes: {num_changes}")
    print(f"Normalized distance: {normalized_distance:.1%}")
    print("JsonPatch operations:")
    for op in patch.patch:
        print(f"  {op['op']} {op['path']}: {op.get('value', 'N/A')}")
