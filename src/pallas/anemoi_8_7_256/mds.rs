use super::BigInteger256;
use super::Felt;
use super::NUM_COLUMNS;

/// Maximum Diffusion Layer matrix for Anemoi
/// [ 1  6  5  5]
/// [25 30  6 11]
/// [25 25  1  6]
/// [ 6 11  5  6]
#[allow(unused)]
pub(crate) const MDS: [Felt; NUM_COLUMNS * NUM_COLUMNS] = [
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xa1a55e68ffffffed,
        0x74c2a54b4f4982f3,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xa1a55e68ffffffed,
        0x74c2a54b4f4982f3,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc3861458ffffff9d,
        0xbeb2d688673baa53,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xcbfe41d4ffffff89,
        0x112ee2d7ad38342b,
        0xfffffffffffffff0,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x4568c830ffffffd5,
        0x3e244daa7012285d,
        0xfffffffffffffffa,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc3861458ffffff9d,
        0xbeb2d688673baa53,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xc3861458ffffff9d,
        0xbeb2d688673baa53,
        0xfffffffffffffff2,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x34786d38fffffffd,
        0x992c350be41914ad,
        0xffffffffffffffff,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x4568c830ffffffd5,
        0x3e244daa7012285d,
        0xfffffffffffffffa,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0xa1a55e68ffffffed,
        0x74c2a54b4f4982f3,
        0xfffffffffffffffd,
        0x3fffffffffffffff,
    ])),
    Felt::new(BigInteger256([
        0x3cf09ab4ffffffe9,
        0xeba8415b2a159e85,
        0xfffffffffffffffc,
        0x3fffffffffffffff,
    ])),
];
