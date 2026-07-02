use super::audio::samples_to_ms;
use super::segment::expand_bounds;

#[test]
fn expands_with_pre_roll_and_tail_padding() {
    let (start, end) = expand_bounds(1_000, 3_000, 200, 300, 4_000);
    assert_eq!((start, end), (800, 3_300));
}

#[test]
fn expansion_stays_inside_available_audio() {
    let (start, end) = expand_bounds(100, 3_000, 500, 2_000, 3_200);
    assert_eq!((start, end), (0, 3_200));
}

#[test]
fn samples_to_ms_uses_sample_rate() {
    assert_eq!(samples_to_ms(8_000, 16_000), 500);
}
