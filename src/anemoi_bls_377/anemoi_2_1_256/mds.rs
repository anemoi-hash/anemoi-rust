use super::BigInteger384;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
// [1, 2]
// [2, 1]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
    ])),
    Felt::new(BigInteger384([
        0x059bfffffffffed0,
        0xa2813f06ffffff62,
        0x3efb675314fa7fe4,
        0xf69d2f6edcf8c60b,
        0x99e92b7f007909d0,
        0x011accc3c5fbe934,
    ])),
    Felt::new(BigInteger384([
        0x059bfffffffffed0,
        0xa2813f06ffffff62,
        0x3efb675314fa7fe4,
        0xf69d2f6edcf8c60b,
        0x99e92b7f007909d0,
        0x011accc3c5fbe934,
    ])),
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
    ])),
];
