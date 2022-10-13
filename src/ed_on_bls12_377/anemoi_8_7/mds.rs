use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [  1  23  22  22]
/// [484 506  23  45]
/// [484 484   1  23]
/// [ 23  45  22  23]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
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
        0x025dffffffffe614,
        0xb13b6ac83fffe50f,
        0x3e40f1018c799cff,
        0x0d184fbb82b224ed,
    ])),
    Felt::new(BigInteger256([
        0x21b8ffffffffe4e6,
        0xea230a2edfffe3d5,
        0x298df730c2c7fb67,
        0x06e6ba21a2071907,
    ])),
    Felt::new(BigInteger256([
        0x9c777ffffffffec5,
        0xab3f94760ffffeb8,
        0x02251ba4877a6e56,
        0x071a44984b108eb7,
    ])),
    Felt::new(BigInteger256([
        0xbbd27ffffffffd97,
        0xe42733dcaffffd7e,
        0xed7221d3bdc8ccbe,
        0x00e8aefe6a6582d0,
    ])),
    Felt::new(BigInteger256([
        0x025dffffffffe614,
        0xb13b6ac83fffe50f,
        0x3e40f1018c799cff,
        0x0d184fbb82b224ed,
    ])),
    Felt::new(BigInteger256([
        0x025dffffffffe614,
        0xb13b6ac83fffe50f,
        0x3e40f1018c799cff,
        0x0d184fbb82b224ed,
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
        0x9c777ffffffffec5,
        0xab3f94760ffffeb8,
        0x02251ba4877a6e56,
        0x071a44984b108eb7,
    ])),
    Felt::new(BigInteger256([
        0xbbd27ffffffffd97,
        0xe42733dcaffffd7e,
        0xed7221d3bdc8ccbe,
        0x00e8aefe6a6582d0,
    ])),
    Felt::new(BigInteger256([
        0x296c7ffffffffed3,
        0x929216656ffffec7,
        0x4c01534d92860e69,
        0x0c79cfc4b9819970,
    ])),
    Felt::new(BigInteger256([
        0x9c777ffffffffec5,
        0xab3f94760ffffeb8,
        0x02251ba4877a6e56,
        0x071a44984b108eb7,
    ])),
];
