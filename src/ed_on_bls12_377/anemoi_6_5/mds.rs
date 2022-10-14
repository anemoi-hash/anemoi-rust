//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [23  1 23]
/// [ 1  1 22]
/// [22  1  1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0x9c777ffffffffec5,
        0xab3f94760ffffeb8,
        0x02251ba4877a6e56,
        0x071a44984b108eb7,
    ])),
    Felt::new(BigInteger256([
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d,
    ])),
    Felt::new(BigInteger256([
        0x9c777ffffffffec5,
        0xab3f94760ffffeb8,
        0x02251ba4877a6e56,
        0x071a44984b108eb7,
    ])),
    Felt::new(BigInteger256([
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d,
    ])),
    Felt::new(BigInteger256([
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d,
    ])),
    Felt::new(BigInteger256([
        0x296c7ffffffffed3,
        0x929216656ffffec7,
        0x4c01534d92860e69,
        0x0c79cfc4b9819970,
    ])),
    Felt::new(BigInteger256([
        0x296c7ffffffffed3,
        0x929216656ffffec7,
        0x4c01534d92860e69,
        0x0c79cfc4b9819970,
    ])),
    Felt::new(BigInteger256([
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d,
    ])),
    Felt::new(BigInteger256([
        0x7d1c7ffffffffff3,
        0x7257f50f6ffffff2,
        0x16d81575512c0fee,
        0x0d4bda322bbb9a9d,
    ])),
];
