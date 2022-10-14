//! MDS matrix implementation for Anemoi

use super::BigInteger384;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [  1  15]
/// [ 15 256]
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
        0x05547fffffff7986,
        0x11c3dc611fffba1e,
        0xda9e39e07be3a3e5,
        0x4d4eefb142f7c397,
        0xa2dc896fcece2a27,
        0x00778a27853b0c5a,
    ])),
];
