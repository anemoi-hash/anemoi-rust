//! MDS matrix implementation for Anemoi

use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [1, 22]
/// [22, 485]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
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
        0x7568ffffffffe606,
        0xc9e8e8d8dfffe500,
        0xf464b958816dfcec,
        0x07b8c48f14411a33,
    ])),
];
