use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi.
/// [1, 3]
/// [3, 10]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0xd35d438dc58f0d9d,
        0x0a78eb28f5c70b3d,
        0x666ea36f7879462c,
        0x0e0a77c19a07df2f,
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
        0xc9638b5c069c8d94,
        0x39b65a76c8e2db4f,
        0x8fb1d6edb1ba0cfd,
        0x2ba010aa41eb7786,
    ])),
];
