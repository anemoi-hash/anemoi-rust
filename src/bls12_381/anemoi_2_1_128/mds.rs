use super::BigInteger384;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [1  2]
/// [2  5]
#[allow(unused)]
pub(crate) const MDS: [Felt; (NUM_COLUMNS + 1) * (NUM_COLUMNS + 1)] = [
    Felt::new(BigInteger384([
        0x760900000002fffd,
        0xebf4000bc40c0002,
        0x5f48985753c758ba,
        0x77ce585370525745,
        0x5c071a97a256ec6d,
        0x15f65ec3fa80e493,
    ])),
    Felt::new(BigInteger384([
        0x321300000006554f,
        0xb93c0018d6c40005,
        0x57605e0db0ddbb51,
        0x8b256521ed1f9bcb,
        0x6cf28d7901622c03,
        0x11ebab9dbb81e28c,
    ])),
    Felt::new(BigInteger384([
        0x321300000006554f,
        0xb93c0018d6c40005,
        0x57605e0db0ddbb51,
        0x8b256521ed1f9bcb,
        0x6cf28d7901622c03,
        0x11ebab9dbb81e28c,
    ])),
    Felt::new(BigInteger384([
        0x6631000000105545,
        0x211400400eec000d,
        0x3fa7af30c820e316,
        0xc52a8b8d6387695d,
        0x9fb4e61d1e83eac5,
        0x05cb922afe84dc77,
    ])),
];
