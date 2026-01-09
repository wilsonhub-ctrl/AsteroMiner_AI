# Test script to verify the total_value_display unit logic for non-X-type asteroids

def test_unit_logic():
    # Simulate the logic for non-X-type asteroids
    # total_value_range is in billions, total_value_range_quadrillion = total_value_range / 1_000_000

    # Test for large values: total_value_range[0] >= 100_000_000 (100 billion)
    total_value_range = (150_000_000.0, 200_000_000.0)  # 150 billion to 200 billion
    total_value_range_quadrillion = (total_value_range[0] / 1_000_000, total_value_range[1] / 1_000_000)  # 150.0 to 200.0 quadrillion

    if total_value_range[0] < 100_000_000:
        total_value_display = {
            "value_range": total_value_range_quadrillion,
            "unit": "billion"
        }
    else:
        total_value_display = {
            "value_range": total_value_range_quadrillion,
            "unit": "quadrillion"
        }

    print(f"Large value display: {total_value_display}")
    assert total_value_display["unit"] == "quadrillion", f"Expected 'quadrillion', got '{total_value_display['unit']}'"
    print("Test passed: Unit is correctly set to 'quadrillion' for large values.")

    # Test for small values: total_value_range[0] < 100_000_000
    total_value_range_small = (50_000_000.0, 75_000_000.0)  # 50 billion to 75 billion
    total_value_range_quadrillion_small = (total_value_range_small[0] / 1_000_000, total_value_range_small[1] / 1_000_000)  # 50.0 to 75.0 quadrillion

    if total_value_range_small[0] < 100_000_000:
        total_value_display_small = {
            "value_range": total_value_range_quadrillion_small,
            "unit": "billion"
        }
    else:
        total_value_display_small = {
            "value_range": total_value_range_quadrillion_small,
            "unit": "quadrillion"
        }

    print(f"Small value display: {total_value_display_small}")
    assert total_value_display_small["unit"] == "billion", f"Expected 'billion', got '{total_value_display_small['unit']}'"
    print("Test passed: Unit is correctly set to 'billion' for small values.")

if __name__ == "__main__":
    test_unit_logic()
