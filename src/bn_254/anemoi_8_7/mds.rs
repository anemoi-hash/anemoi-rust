use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [ 1  4  3  3]
/// [ 9 12  4  7]
/// [ 9  9  1  4]
/// [ 4  7  3  4]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0xd35d438dc58f0d9d,
        0x0a78eb28f5c70b3d,
        0x666ea36f7879462c,
        0x0e0a77c19a07df2f,
    ])),
    Felt::new(BigInteger256([
        0x115482203dbf392d,
        0x926242126eaa626a,
        0xe16a48076063c052,
        0x07c5909386eddc93,
    ])),
    Felt::new(BigInteger256([
        0x7a17caa950ad28d7,
        0x1f6ac17ae15521b9,
        0x334bea4e696bd284,
        0x2a1f6744ce179d8e,
    ])),
    Felt::new(BigInteger256([
        0x7a17caa950ad28d7,
        0x1f6ac17ae15521b9,
        0x334bea4e696bd284,
        0x2a1f6744ce179d8e,
    ])),
    Felt::new(BigInteger256([
        0xf60647ce410d7ff7,
        0x2f3d6f4dd31bd011,
        0x2943337e3940c6d1,
        0x1d9598e8a7e39857,
    ])),
    Felt::new(BigInteger256([
        0x33fd8660b93dab87,
        0xb726c6374bff273e,
        0xa43ed816212b40f7,
        0x1750b1ba94c995bb,
    ])),
    Felt::new(BigInteger256([
        0x115482203dbf392d,
        0x926242126eaa626a,
        0xe16a48076063c052,
        0x07c5909386eddc93,
    ])),
    Felt::new(BigInteger256([
        0x4f4bc0b2b5ef64bd,
        0x1a4b98fbe78db996,
        0x5c65ec9f484e3a79,
        0x0180a96573d3d9f8,
    ])),
    Felt::new(BigInteger256([
        0xf60647ce410d7ff7,
        0x2f3d6f4dd31bd011,
        0x2943337e3940c6d1,
        0x1d9598e8a7e39857,
    ])),
    Felt::new(BigInteger256([
        0xf60647ce410d7ff7,
        0x2f3d6f4dd31bd011,
        0x2943337e3940c6d1,
        0x1d9598e8a7e39857,
    ])),
    Felt::new(BigInteger256([
        0xd35d438dc58f0d9d,
        0x0a78eb28f5c70b3d,
        0x666ea36f7879462c,
        0x0e0a77c19a07df2f,
    ])),
    Felt::new(BigInteger256([
        0x115482203dbf392d,
        0x926242126eaa626a,
        0xe16a48076063c052,
        0x07c5909386eddc93,
    ])),
    Felt::new(BigInteger256([
        0x115482203dbf392d,
        0x926242126eaa626a,
        0xe16a48076063c052,
        0x07c5909386eddc93,
    ])),
    Felt::new(BigInteger256([
        0x4f4bc0b2b5ef64bd,
        0x1a4b98fbe78db996,
        0x5c65ec9f484e3a79,
        0x0180a96573d3d9f8,
    ])),
    Felt::new(BigInteger256([
        0x7a17caa950ad28d7,
        0x1f6ac17ae15521b9,
        0x334bea4e696bd284,
        0x2a1f6744ce179d8e,
    ])),
    Felt::new(BigInteger256([
        0x115482203dbf392d,
        0x926242126eaa626a,
        0xe16a48076063c052,
        0x07c5909386eddc93,
    ])),
];
