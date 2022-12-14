//! MDS matrix implementation for Anemoi

use super::BigInteger384;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [3 1 3]
/// [1 1 2]
/// [2 1 1]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger384([
        0xee1d00000009aaa1,
        0x86840025e97c0007,
        0x4f7823c40df41de8,
        0x9e7c71f069ece051,
        0x7dde005a606d6b99,
        0x0de0f8777c82e085,
    ])),
    Felt::new(BigInteger384([
        0x760900000002fffd,
        0xebf4000bc40c0002,
        0x5f48985753c758ba,
        0x77ce585370525745,
        0x5c071a97a256ec6d,
        0x15f65ec3fa80e493,
    ])),
    Felt::new(BigInteger384([
        0xee1d00000009aaa1,
        0x86840025e97c0007,
        0x4f7823c40df41de8,
        0x9e7c71f069ece051,
        0x7dde005a606d6b99,
        0x0de0f8777c82e085,
    ])),
    Felt::new(BigInteger384([
        0x760900000002fffd,
        0xebf4000bc40c0002,
        0x5f48985753c758ba,
        0x77ce585370525745,
        0x5c071a97a256ec6d,
        0x15f65ec3fa80e493,
    ])),
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
        0x760900000002fffd,
        0xebf4000bc40c0002,
        0x5f48985753c758ba,
        0x77ce585370525745,
        0x5c071a97a256ec6d,
        0x15f65ec3fa80e493,
    ])),
    Felt::new(BigInteger384([
        0x760900000002fffd,
        0xebf4000bc40c0002,
        0x5f48985753c758ba,
        0x77ce585370525745,
        0x5c071a97a256ec6d,
        0x15f65ec3fa80e493,
    ])),
];
