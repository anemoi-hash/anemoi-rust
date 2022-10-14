//! MDS matrix implementation for Anemoi

use super::BigInteger384;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [16  1 16]
/// [ 1  1 15]
/// [15  1  1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger384([
        0x93b43ffffffff67b,
        0xa0d125e30ffffb0d,
        0x5d1a4faa05a59724,
        0x323b39b7e2fcce8e,
        0xf0223f35e4a1e060,
        0x006f42bfb905f50e,
    ])),
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
    ])),
    Felt::new(BigInteger384([
        0x93b43ffffffff67b,
        0xa0d125e30ffffb0d,
        0x5d1a4faa05a59724,
        0x323b39b7e2fcce8e,
        0xf0223f35e4a1e060,
        0x006f42bfb905f50e,
    ])),
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
    ])),
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
    ])),
    Felt::new(BigInteger384([
        0x15eefffffffff714,
        0x669be3a3bffffb5d,
        0xdc8ffe3035319f32,
        0xd10f7bf375757f17,
        0x6968af36d106a4b2,
        0x019016a3edcd115f,
    ])),
    Felt::new(BigInteger384([
        0x15eefffffffff714,
        0x669be3a3bffffb5d,
        0xdc8ffe3035319f32,
        0xd10f7bf375757f17,
        0x6968af36d106a4b2,
        0x019016a3edcd115f,
    ])),
    Felt::new(BigInteger384([
        0x02cdffffffffff68,
        0x51409f837fffffb1,
        0x9f7db3a98a7d3ff2,
        0x7b4e97b76e7c6305,
        0x4cf495bf803c84e8,
        0x008d6661e2fdf49a,
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
