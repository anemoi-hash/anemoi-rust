//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                apply_permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            apply_permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        apply_permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        // We can output as few as 2 elements while
        // maintaining the targeted security level.
        assert!(k <= NUM_COLUMNS);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger256([
                0x20c3e5eca821293a,
                0xd62b99a543d512a4,
                0x7acf48c392e7c9cf,
                0x140b174a850d2d0e,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x6136484f272ce3cf,
                    0x5b11d211255574d7,
                    0x06817557243d9c6d,
                    0x0aab2750b583e358,
                ])),
                Felt::new(BigInteger256([
                    0xe111fda206c60be3,
                    0x075892a2e1cefef1,
                    0x39d69e7c65e45f45,
                    0x5f255b0618eba761,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9525ff48b08fdb2f,
                    0x53f94b742eb4084d,
                    0xa5beb7c7a5850709,
                    0x31ccd1e5190c7aac,
                ])),
                Felt::new(BigInteger256([
                    0x5bfd64726e002cbd,
                    0xce40e337c3c6f3f9,
                    0xd2a0b8b5aeb39312,
                    0x0a04e6076d8d96e6,
                ])),
                Felt::new(BigInteger256([
                    0x15cfde90039db515,
                    0x66a92b0924faff18,
                    0xd70e165cabef77b8,
                    0x2117f36d945a8f5f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xec4fd020d26bdc24,
                    0x51c2569a0cd0de57,
                    0x96a61aba4bdd1ee4,
                    0x5c321a985750ea10,
                ])),
                Felt::new(BigInteger256([
                    0xf7a5b5dda5ddd0a2,
                    0xcfb1201dbc3991da,
                    0xc3e17cc8710f497c,
                    0x5434e718aa4be6bd,
                ])),
                Felt::new(BigInteger256([
                    0x0db0268abf14506d,
                    0x104505924ba016a4,
                    0x7291154a0139bfd8,
                    0x54d2ba53399fa593,
                ])),
                Felt::new(BigInteger256([
                    0x7361ab33996f9f69,
                    0x5bf08e734864391e,
                    0x9add30e44dc49c46,
                    0x235c6cd06b7eea0c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x87f0fe764cec356b,
                    0x8e35a9ed2a316e75,
                    0x51979809e374bbb6,
                    0x4879ddb233447b27,
                ])),
                Felt::new(BigInteger256([
                    0x6bf4a8365221c97e,
                    0x4858fad51cea8c30,
                    0x7c41b3b3dd283c6c,
                    0x53c614b019e61e5c,
                ])),
                Felt::new(BigInteger256([
                    0xbb1beeefdca3b87c,
                    0x84a851f35c6ba412,
                    0x6acc6c1f09b1010e,
                    0x19620fec07370fa4,
                ])),
                Felt::new(BigInteger256([
                    0x48abdf794b9868c7,
                    0x1d4b2ab30ab4f240,
                    0xd9ecb00df4782713,
                    0x670739cbb8c0be87,
                ])),
                Felt::new(BigInteger256([
                    0x52404c8cb4313d34,
                    0x1afc4659e31b5e8c,
                    0xafe264ddc90682b8,
                    0x70dc53d52086d855,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x58b569003e3353fd,
                    0x7f32174736880a83,
                    0x200be3d059f9d14f,
                    0x4639402ff51cc8de,
                ])),
                Felt::new(BigInteger256([
                    0xa1783eaf1f77dbcc,
                    0xa79883aab6478c34,
                    0x21a48a022a6c73a4,
                    0x14d165a4a3e17a86,
                ])),
                Felt::new(BigInteger256([
                    0xe5322d719cb0f646,
                    0x6e930bf2c5ba786b,
                    0xf35cd43368cb48e8,
                    0x0fcaa8066f60e5b6,
                ])),
                Felt::new(BigInteger256([
                    0x711e5ee26c0bf0cc,
                    0x81dbed5eac03c9bc,
                    0xffd3ee33754a291f,
                    0x2bf04ee0b4794f2a,
                ])),
                Felt::new(BigInteger256([
                    0x90cfb711480da654,
                    0xb5b801722aa81f9f,
                    0x56af688c42183f25,
                    0x2b322863d2b6ae36,
                ])),
                Felt::new(BigInteger256([
                    0xc4207ffeca637e4f,
                    0x4d2d703746313cdc,
                    0x65e6af87216907e1,
                    0x64b7671200c66f7e,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe8528f7f1d2cd1fb,
                    0x931e948c6d699f3c,
                    0xf09c412f21b102a3,
                    0x62f774d863cef0d6,
                ])),
                Felt::new(BigInteger256([
                    0xdd640cda26a06d7a,
                    0xb906f56e1ffbd058,
                    0x7dc1ded0303f4216,
                    0x3d0f43d5cb4c32aa,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x17d5094191da66e4,
                    0x3b32a43e32e635b6,
                    0x7c8d8917b2ae27ef,
                    0x1f93097d189e135f,
                ])),
                Felt::new(BigInteger256([
                    0x0753a5126f03db0c,
                    0x033cc1b030e3fda4,
                    0x3de872435cec3547,
                    0x09f1f162840882c8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0f83a74b66fef712,
                    0x4ed366e75b106700,
                    0x2698c00fac29a0c8,
                    0x070af381ce958e1c,
                ])),
                Felt::new(BigInteger256([
                    0x79c6b82c2adfba18,
                    0x476689574d0f8c2a,
                    0xce58280c40ad36b7,
                    0x2d7a11f092e2bd3b,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4a01c786c08f23ad,
                    0x52e90e1598147aee,
                    0x3f366c3d6a984cf5,
                    0x691c9204c06a994e,
                ])),
                Felt::new(BigInteger256([
                    0x362e717c92396c7e,
                    0x36db430c833ec335,
                    0x7b1b47be8c1b6117,
                    0x282ba00252ec6340,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe94f9665071d8d1d,
                    0x1ad36f9c439bba4c,
                    0x4f5d64ec0c4b736d,
                    0x6d011206d58a0154,
                ])),
                Felt::new(BigInteger256([
                    0x5d3283f4d64a8172,
                    0x03979fa1b60d8654,
                    0x36651d874bf23f03,
                    0x4417822439c52fe9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x5d2e9298f83248b6,
                    0xd53c36ce342599db,
                    0x3b7045722a62d833,
                    0x3201bb6ad72a38f1,
                ])),
                Felt::new(BigInteger256([
                    0x6559e1f967918fa1,
                    0x2d2a6798c63916f7,
                    0x6a6553b58fb35f77,
                    0x6263d3d1b51f1f88,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x77c12d4943fe672a,
                    0xbd870f4f347d9d3a,
                    0x3ddb331fb4fc4e86,
                    0x3e24b094e813d88a,
                ])),
                Felt::new(BigInteger256([
                    0xdb8efd2f83a1b8c5,
                    0xd86c91a803f7f035,
                    0x39bc78f9ce66dbe9,
                    0x52c6a05831b31982,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x22fc8acbe504c453,
                    0x1cfacf222e44d1d4,
                    0xeec19b110749c9fe,
                    0x19cc400faa3e6356,
                ])),
                Felt::new(BigInteger256([
                    0x7b8b1f433e49ec3f,
                    0x83137c1f3714d2a0,
                    0x9cc5ded4779eb289,
                    0x5c7d7fe689f36859,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6358d4420f54b14a,
                    0xe48e41d31eb7f4cc,
                    0x5c456e557f652256,
                    0x52c432056a4ea968,
                ])),
                Felt::new(BigInteger256([
                    0x5f017af59871436a,
                    0x89368db47437b406,
                    0x2edeff0a757b07f8,
                    0x6af74b328dab9189,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb2c054b700092a45,
                    0x1d9cd394dafa1ef2,
                    0x69d4379b3aa6916d,
                    0x6f8fb3c463e6e679,
                ])),
                Felt::new(BigInteger256([
                    0xb30cda1b48224bef,
                    0xea1b486ea115cad1,
                    0x3e7a516ef1c87902,
                    0x060f52567a1753fd,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5498bb22694106a7,
                    0xe4556cf5b738630d,
                    0xccd5a22b48183db2,
                    0x4936e2db0611a2e5,
                ])),
                Felt::new(BigInteger256([
                    0x2dc5e74bf832b9cd,
                    0x1e8696467f89d7f6,
                    0x0d15af45fe5c9304,
                    0x34cc9389a6ea50b9,
                ])),
                Felt::new(BigInteger256([
                    0xdc2b75acb78a4154,
                    0x9a84d246622ebd2a,
                    0x7e47da6e3481dda2,
                    0x2b8c58da7f94a774,
                ])),
                Felt::new(BigInteger256([
                    0x19e589fb4bc1379f,
                    0x8ea7824bb4a1b26f,
                    0x6bf899183e091bc6,
                    0x160c895f2d11c89f,
                ])),
                Felt::new(BigInteger256([
                    0x7b4bb85b9d58e552,
                    0x54c6533bdffe4fa7,
                    0x540e64f715bb1797,
                    0x58609e5aff52c416,
                ])),
                Felt::new(BigInteger256([
                    0x5d3f41fc84549142,
                    0xbe4c8faa6332d823,
                    0x27396a1029076fc4,
                    0x44527e326cce4c97,
                ])),
                Felt::new(BigInteger256([
                    0x51fe7841a977f943,
                    0xea0270bce1b682d2,
                    0x3fb8913e57660968,
                    0x04e1385960c28f6c,
                ])),
                Felt::new(BigInteger256([
                    0xb8d617a7cdf640df,
                    0xb3657f9ad49419d3,
                    0xeb6fa80f2f0ee521,
                    0x0c918d51f0eb58b3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0cd9c5f185e940d7,
                    0x64084547aebc5bc6,
                    0x27d770a00ba9e811,
                    0x4155d491f7ba62c1,
                ])),
                Felt::new(BigInteger256([
                    0x955ecd12ad85409a,
                    0x4379d12becb96db8,
                    0x9e4ecc03af77f6d5,
                    0x344c72806c1d918d,
                ])),
                Felt::new(BigInteger256([
                    0xb95b178a94646ea6,
                    0x243737310eaa1678,
                    0x968e60b6ca134c77,
                    0x4dd9ef60db692baa,
                ])),
                Felt::new(BigInteger256([
                    0xf615dde8763cf5fe,
                    0x9f6e236aa061ee97,
                    0x6d814db40bde4a7f,
                    0x645681d5d101693a,
                ])),
                Felt::new(BigInteger256([
                    0xbff10725cf2dcc11,
                    0x59d0f1ca36723b3e,
                    0x7118296a214ec762,
                    0x3b9c9387f56602a5,
                ])),
                Felt::new(BigInteger256([
                    0x43ee9857506841cd,
                    0x6d4d93f21fa24b79,
                    0x6932d59fda647895,
                    0x22e7283572d2ba6a,
                ])),
                Felt::new(BigInteger256([
                    0xb9c024c5a71673c0,
                    0x6dc09044dafe038c,
                    0xca413caa39cf298b,
                    0x08bb81d3d9637270,
                ])),
                Felt::new(BigInteger256([
                    0x8f5cca0ae220523e,
                    0x873b6b8737ef24bb,
                    0x280548a84999a3fe,
                    0x6fa4234f68fcc5d0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5e0c769d373c5da4,
                    0xc718650b88515dc6,
                    0xd809349a82d2724e,
                    0x5293712fd6bca6f0,
                ])),
                Felt::new(BigInteger256([
                    0x222ad460a787fad6,
                    0x5e7e1b16b233faa0,
                    0x3f8badfc361dc80f,
                    0x523a1caa7d95a66e,
                ])),
                Felt::new(BigInteger256([
                    0x86e22ad842cbc885,
                    0x850fc05980201737,
                    0xd5d016f5ad921450,
                    0x51132e880ded39fa,
                ])),
                Felt::new(BigInteger256([
                    0xfedfe30ae3d6951e,
                    0x5c471a7851ddce70,
                    0xa45ece9e1fae602e,
                    0x545b7d008a859786,
                ])),
                Felt::new(BigInteger256([
                    0xb39449cd1b662dc8,
                    0xc35339c055006534,
                    0x5c002d506936c14c,
                    0x070cb32dea4420e1,
                ])),
                Felt::new(BigInteger256([
                    0xaf20264d7e53c69c,
                    0x95b74320bb7954a9,
                    0xf5b3dccf87a8d3e4,
                    0x613354b1faa3624c,
                ])),
                Felt::new(BigInteger256([
                    0x7d430794a5305054,
                    0x4d8c3cd5fc743e27,
                    0x357eacf8f2d1bb68,
                    0x478b8370b994232e,
                ])),
                Felt::new(BigInteger256([
                    0xff343ff0a2a0ab3a,
                    0x932c53ad0f898a44,
                    0x0e55ea86fd746ca9,
                    0x1311bfef25980f55,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7cd55d6a801dfa80,
                    0x5b8e09cd3d4ebdf1,
                    0x3ec4e5bdb1552f19,
                    0x73431d86cc92e157,
                ])),
                Felt::new(BigInteger256([
                    0x6bc4e255b60404e2,
                    0x58b80591de4444d0,
                    0x3cda2ffcf9d1e95d,
                    0x30e9014d9d258869,
                ])),
                Felt::new(BigInteger256([
                    0x4c154390b27f012f,
                    0xbc06983fb9cb407b,
                    0x1d37ef669161fbee,
                    0x14720f6d41b8f918,
                ])),
                Felt::new(BigInteger256([
                    0xa092bbd11f20f213,
                    0x0bed7d7b73f65ef7,
                    0x97558da9a81f51d2,
                    0x3ef150e676cb17ea,
                ])),
                Felt::new(BigInteger256([
                    0xea6cbda9d7d9422e,
                    0x365ec8ca7687188f,
                    0xd0f21fcdc7501c94,
                    0x2d7d1c75ad76609e,
                ])),
                Felt::new(BigInteger256([
                    0xf34f12203d3f52b8,
                    0xecc39ecf553a24c1,
                    0x40325f508cdd23a1,
                    0x197fb2b38a4d5aff,
                ])),
                Felt::new(BigInteger256([
                    0x8318b23866b06b37,
                    0x8fca02f0ee7eab2d,
                    0xc1424a0c027cfe0a,
                    0x3eb972b828597a90,
                ])),
                Felt::new(BigInteger256([
                    0xf9e4c74650ad8b9e,
                    0x3be9da34d7efffc9,
                    0x3c17ea7e1f9b2bc6,
                    0x6f2a124d1270da53,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6021fe8ee694eff5,
                    0x615f7bcad78b8cbb,
                    0x81d479f20b47d9ff,
                    0x2f5b692ddfb86b11,
                ])),
                Felt::new(BigInteger256([
                    0x06a82fccff36a546,
                    0x75a70915e09dd640,
                    0x03f8fe4fd68df10e,
                    0x2bad898b8005c6ae,
                ])),
                Felt::new(BigInteger256([
                    0x9bf402ad78fbb8c1,
                    0xd0617df3c4e564fa,
                    0x9b862ace8f154d22,
                    0x25971e29d10fa254,
                ])),
                Felt::new(BigInteger256([
                    0x907238207be3717d,
                    0x1a777657c7c404fe,
                    0x6e97559ce9a5ecdc,
                    0x3dcaa4c43ec52b8d,
                ])),
                Felt::new(BigInteger256([
                    0x457c203b3d5571ac,
                    0x7b66d7e2174b73fc,
                    0x8d5b137ed86e3fb9,
                    0x1a5455afc66d467f,
                ])),
                Felt::new(BigInteger256([
                    0x50d83e2b3ba9e53d,
                    0x4423d996757ee1ff,
                    0xb59e0886e3999f69,
                    0x3106011c4467c1b7,
                ])),
                Felt::new(BigInteger256([
                    0x20cc2d2a0e0d461c,
                    0x309a9a578a6a0f21,
                    0x7a7f7de26fc54ece,
                    0x0797db8bcda3d726,
                ])),
                Felt::new(BigInteger256([
                    0xfec3ca570afce4c2,
                    0x91d5188f74b7b6a1,
                    0x655683fbd1c74a47,
                    0x17a8e35311c39095,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xef36513151c39041,
                    0xe332b4c78dc193dc,
                    0x051c9f03063c72f2,
                    0x3cba4f2dfbb1a30e,
                ])),
                Felt::new(BigInteger256([
                    0xef889fe9888b5909,
                    0x68988dd34c34e264,
                    0x4b7571c3b7f324a2,
                    0x35aaa13141707781,
                ])),
                Felt::new(BigInteger256([
                    0x551e554d32d66e4a,
                    0x3bf282753b14e7ce,
                    0x268f40344314930f,
                    0x7210611e4e7979bf,
                ])),
                Felt::new(BigInteger256([
                    0x02ce5cc16091e6da,
                    0xf2af5b12fe71c90f,
                    0x1f096efc7a869546,
                    0x05ffea875a898a08,
                ])),
                Felt::new(BigInteger256([
                    0x4da6746f30d47871,
                    0x24a76323c96ceafa,
                    0x357baa9c4df1bd6a,
                    0x46d77cbdcb538961,
                ])),
                Felt::new(BigInteger256([
                    0x632808e3675ffcc1,
                    0x3178c3f9aab3a0c2,
                    0x4a023b18d1544e23,
                    0x3ca26eec178266f7,
                ])),
                Felt::new(BigInteger256([
                    0xd1c86d5c1dc25e00,
                    0xee17691c95238818,
                    0x820df2b52aa67278,
                    0x64719a877e079249,
                ])),
                Felt::new(BigInteger256([
                    0x05e89da2354f0a1e,
                    0xfb6b86fe8f88de43,
                    0x3db75f933b8085b8,
                    0x351ae6c6e25513d6,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x638941f7c690ddef,
                    0x1ed0d407bb46e38f,
                    0x49d5d0e3587c7b63,
                    0x5f8ab027800a9ca7,
                ])),
                Felt::new(BigInteger256([
                    0x766092d500d0ada9,
                    0xbfeedb8d5cb8f199,
                    0x771d3319220d0386,
                    0x1c079d1513783315,
                ])),
                Felt::new(BigInteger256([
                    0x1db86b47d2305016,
                    0x141acdf68087a599,
                    0xeeadb7808142b96c,
                    0x61a831e8ef7289a5,
                ])),
                Felt::new(BigInteger256([
                    0xd67aedd697228559,
                    0x19c3ba8471b8384a,
                    0x3ec34d2ef3dc4761,
                    0x378dd63557158ba9,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa18125a649f7d529,
                    0x3b3ef9d86424719c,
                    0x0d63b3cf91e08ead,
                    0x2716a1aba2466996,
                ])),
                Felt::new(BigInteger256([
                    0x3d2e13cfe2c3deec,
                    0x7e0f64a2dd2ea965,
                    0x2f7423a1dfc2dca2,
                    0x0ceb76c20defc170,
                ])),
                Felt::new(BigInteger256([
                    0x88482c528888b94b,
                    0x24c6f30e196e4a8a,
                    0xb676916d04b193c5,
                    0x525312bfdfd231ac,
                ])),
                Felt::new(BigInteger256([
                    0xb683b1cd22e5d0ce,
                    0x6e800dbac23eb798,
                    0x57c9bd094b401be8,
                    0x06492e35d0a13a28,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x45e868729a6928f7,
                    0x6d8ff3beb4e345e8,
                    0x9fdf62e21436208e,
                    0x0f8b11a9420ce9d7,
                ])),
                Felt::new(BigInteger256([
                    0x78380dafc6020ccd,
                    0xdd920bcec3ba1cae,
                    0x83a4d2342847c077,
                    0x1be6b685ccf73dae,
                ])),
                Felt::new(BigInteger256([
                    0xa9edf472b6c45897,
                    0x17ede95831bdbe10,
                    0xb88e5beda6711b25,
                    0x49b47af24b94b54a,
                ])),
                Felt::new(BigInteger256([
                    0xb5a11ec79bc4527b,
                    0xd5bc80696a28ca30,
                    0x8185045361614091,
                    0x13073a3ddf435fd3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7f78612dc7b0e766,
                    0x4f7b6d078210aa3b,
                    0x24e6a0be2edff11d,
                    0x131573eefb96794c,
                ])),
                Felt::new(BigInteger256([
                    0x005d4f51a197bb54,
                    0xcdbb8a1ef7459380,
                    0xed21607a07d80a34,
                    0x69f6789c150c354f,
                ])),
                Felt::new(BigInteger256([
                    0xb23a950109367323,
                    0xb0d8b064fe2b7a50,
                    0x0aeb93ab76f99781,
                    0x3d1f963234a539b2,
                ])),
                Felt::new(BigInteger256([
                    0xd7713051e6756a27,
                    0x43d28b73eec71ae7,
                    0xd41a382440f6f915,
                    0x61f55e44519c3825,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x2f0dbae9080b77f1,
                    0x2670689b82264e58,
                    0x5f8c1b8497598adb,
                    0x37604c37e7cbb6c0,
                ])),
                Felt::new(BigInteger256([
                    0x65e17d175394d347,
                    0xf54226dce8e257bd,
                    0x699d0b888826a8a9,
                    0x369189a37e887378,
                ])),
                Felt::new(BigInteger256([
                    0xc8a1b256ccd5c6af,
                    0x792636437b11a2b0,
                    0xf4b0e452c30bcf07,
                    0x106423eced7e179a,
                ])),
                Felt::new(BigInteger256([
                    0xe7b06b5994a8e5f1,
                    0x4dfec181eba51e5d,
                    0xe659fa3384c8d91b,
                    0x26e44293b51278eb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3e5f9068c17684df,
                    0x34fdbb184f24ca25,
                    0xf99d110e94f9c395,
                    0x500504a8ccebc12c,
                ])),
                Felt::new(BigInteger256([
                    0xc8c2ad3f0db7fe42,
                    0xc090bbd0e5adfffa,
                    0xb0a2a2ab1c48f882,
                    0x588840b01c1826a8,
                ])),
                Felt::new(BigInteger256([
                    0x7d48aab9f0fcf3a6,
                    0xc71ad8ddfa865119,
                    0x9ce056adfbdd7f53,
                    0x5a9b81d665052bb7,
                ])),
                Felt::new(BigInteger256([
                    0x597fba93b2dc2b17,
                    0x4db022551dff4620,
                    0xc70923d26729e613,
                    0x3ab6617c5cd04d5d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6a6bb5612237d709,
                    0xf1341906cc3c81d6,
                    0xec06bfb855fc0109,
                    0x2bbeabdcd4a66483,
                ])),
                Felt::new(BigInteger256([
                    0xc9a8bf21345ae031,
                    0xf8b40722ea5d00d4,
                    0x3567964daef4935d,
                    0x2059b7adc6d714b1,
                ])),
                Felt::new(BigInteger256([
                    0x56f534164301f1e8,
                    0xec57f5d911a0e396,
                    0xc3789fe5d4e5c167,
                    0x3e1fd571233f1894,
                ])),
                Felt::new(BigInteger256([
                    0xd7c395c22d301214,
                    0xfecca774e33ba11e,
                    0x0272227363e5c3ea,
                    0x6d98711a3ab155b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x685a450f2b09b21b,
                    0xac8779f2c6b50e54,
                    0xc96d7500625f5901,
                    0x36d35632efbbeb3f,
                ])),
                Felt::new(BigInteger256([
                    0x690a0085653a3602,
                    0x79806d0461a73bf3,
                    0x6f74ddfcc40b3abd,
                    0x605b67a73759ed37,
                ])),
                Felt::new(BigInteger256([
                    0xb6c83c76eff8210d,
                    0x123eeaaa2ed356a8,
                    0xb71b81bc45bc6bd3,
                    0x26f8de11386e8374,
                ])),
                Felt::new(BigInteger256([
                    0x52495580eb7f1c94,
                    0x20f7d58f46ab1c27,
                    0x7756a2effe1024cb,
                    0x599ad4832b9dca69,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7dccd988f2448694,
                    0x632ca94caad4b446,
                    0xf9241500996f05e4,
                    0x16a80b28971531f4,
                ])),
                Felt::new(BigInteger256([
                    0x880385e373831f3c,
                    0x0ff246e8d5b7ee85,
                    0xe2c9fc443ae3f4cd,
                    0x608a616dbd416788,
                ])),
                Felt::new(BigInteger256([
                    0xa2dd712bd5b67ce7,
                    0xd7df2b4e0f9f913a,
                    0x8426ae6ddab826fe,
                    0x4cf8aa525c51b255,
                ])),
                Felt::new(BigInteger256([
                    0xcd94e3d17c64fc01,
                    0x1fcf09f9bde6023d,
                    0x2913c184961f8aff,
                    0x614def27e32caae1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xde14d77e39137510,
                    0x0bc842601b0eafb4,
                    0xe5f46621c6002f16,
                    0x4b2de25933e50f4a,
                ])),
                Felt::new(BigInteger256([
                    0x3b2ed9c53d140d16,
                    0x4c686fc4236cf598,
                    0xcbcf55b5885c6000,
                    0x32993f92a8cbef28,
                ])),
                Felt::new(BigInteger256([
                    0xa31a318ca8d0e5ac,
                    0x8e47659936d0c453,
                    0x25967b8ac1c225cc,
                    0x422f7ec42c98a50d,
                ])),
                Felt::new(BigInteger256([
                    0x8b7c66eea3475a5b,
                    0xf4cb70185eb70958,
                    0x69ab9dcc2ffeae84,
                    0x2ff4c7dcc9b49a5d,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(); 8],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xad60b51680a71d1c,
                    0xbd4d5fb97e64a05d,
                    0x60802ab71d4c0043,
                    0x6d7efb7dfe55b442,
                ])),
                Felt::new(BigInteger256([
                    0x7a1676493a025624,
                    0x0e70335efb7a4343,
                    0xbc1efa026b1e2f61,
                    0x3cbd3fbd98cbfe7b,
                ])),
                Felt::new(BigInteger256([
                    0xf9c018cd57b94f21,
                    0x0efa31c1ca69a305,
                    0xe96b6fbe921d287d,
                    0x0168b939eaa6a0bf,
                ])),
                Felt::new(BigInteger256([
                    0xc1846b6ab7133e06,
                    0x49b6bc4d1e778620,
                    0x73d9206cc515caa5,
                    0x50094366d91fe9a6,
                ])),
                Felt::new(BigInteger256([
                    0x872d7cc35662c264,
                    0xe9aa9489eb7db61e,
                    0x2fb7bce467504c7f,
                    0x25d5a24ba42092cf,
                ])),
                Felt::new(BigInteger256([
                    0xa578825a4e233020,
                    0xd65f66af95d37475,
                    0x86ba973720f90743,
                    0x16e907d19f2b4d6c,
                ])),
                Felt::new(BigInteger256([
                    0x9f472baec8920ee8,
                    0x571bdc2591d1e1a0,
                    0x901b4a2c441a0359,
                    0x01d1af32891c7522,
                ])),
                Felt::new(BigInteger256([
                    0x3a1e7984aa25221a,
                    0xfe41fa43cb269c97,
                    0x021aeb596c661c54,
                    0x6d69bc405b61da06,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbffdfb62ea16fe21,
                    0x8976d62930e3cf1b,
                    0x885b285d65ee53d3,
                    0x58e2036a11e47803,
                ])),
                Felt::new(BigInteger256([
                    0x4ab960a2469c8676,
                    0x31694041b05738f4,
                    0x672e3d828724de64,
                    0x5439947bd0fa66b1,
                ])),
                Felt::new(BigInteger256([
                    0xa1effc1f4b03554f,
                    0xf201b88521cb7e4e,
                    0x02e5681ce4fb3ff5,
                    0x699e4f2a74772117,
                ])),
                Felt::new(BigInteger256([
                    0x341b32212e90aded,
                    0x92986053574a26e3,
                    0x8e7d1af19c00ec75,
                    0x437bdb5cb7f8fc06,
                ])),
                Felt::new(BigInteger256([
                    0xb498824f9e769d1b,
                    0xf249e0ba032783b6,
                    0x1a6b3660b5fddbca,
                    0x23394242ba44205d,
                ])),
                Felt::new(BigInteger256([
                    0x979185e0ea176c08,
                    0xfaf18186f5615bd6,
                    0x78e069e032886692,
                    0x335c9ecd0e7fdaff,
                ])),
                Felt::new(BigInteger256([
                    0x7232d66ec3cf1214,
                    0x70de7853c4b9e7dd,
                    0xee804b11a92a114c,
                    0x6d604e0ffc89e37e,
                ])),
                Felt::new(BigInteger256([
                    0xaedd19b326910b9d,
                    0xcbbac77be6299247,
                    0x52c84e6a807004f2,
                    0x266fc2481f19fb2a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5d79ace32412c05e,
                    0x0f9170af18bef259,
                    0xc50960b0c83f65a6,
                    0x600739ca4392810f,
                ])),
                Felt::new(BigInteger256([
                    0x486071247915570c,
                    0x94043b0429e90180,
                    0x707b1fa13d0747a5,
                    0x6885a27bb796afe4,
                ])),
                Felt::new(BigInteger256([
                    0x4452824570f55471,
                    0xe3180f1c713e1545,
                    0x459aa8081fd2da52,
                    0x69dedce218f9ccf4,
                ])),
                Felt::new(BigInteger256([
                    0x0e107e6d4f3e3028,
                    0xbb13831b5cfd5bf2,
                    0xd1abd75cb103545b,
                    0x4c49ef9bae349bd9,
                ])),
                Felt::new(BigInteger256([
                    0x49b4bb00bf374471,
                    0x0b6d777964c9b141,
                    0xe64f1fc95d529902,
                    0x611c79ca1d39ad74,
                ])),
                Felt::new(BigInteger256([
                    0x625f9c0053fd1b4c,
                    0xd5e69d2afbf617eb,
                    0x9c1d13d5d97a7d81,
                    0x34c1f546afcfdff4,
                ])),
                Felt::new(BigInteger256([
                    0x475992a4b0a090e0,
                    0x434df34810d2cd82,
                    0xf97f5cb51e93d928,
                    0x6389f5c54886b924,
                ])),
                Felt::new(BigInteger256([
                    0x7fb4a2d5b0589917,
                    0x69a9fb0c52f52a50,
                    0x062b44f442bf58be,
                    0x1df8e556e891054f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x7860fc6772a2a938,
                    0x55488da76775e085,
                    0xe700b605b97d0ed2,
                    0x53fe6e1f137d9c7c,
                ])),
                Felt::new(BigInteger256([
                    0x9a6e2cb3a1467cb9,
                    0x8ade021881b34212,
                    0x36e513d57eeffafe,
                    0x6a0f1c11117bc117,
                ])),
                Felt::new(BigInteger256([
                    0xf4e6f629c13c206e,
                    0x50aa38c17d670077,
                    0x30395942ecd434d8,
                    0x5d571ee08dd048c4,
                ])),
                Felt::new(BigInteger256([
                    0xcf1937d218977df4,
                    0x6b8af4b54f6d6063,
                    0x431edbe16b9094b1,
                    0x0157a6ed3eafeff7,
                ])),
                Felt::new(BigInteger256([
                    0x81ac9838fcc854fe,
                    0xe8912ee162349d35,
                    0x73887b1f20d57c83,
                    0x4313af0064f24604,
                ])),
                Felt::new(BigInteger256([
                    0xfea20e71c85e6bf7,
                    0xda62498afaeeab91,
                    0x8d0b849517cb5948,
                    0x1be815d3b6f65572,
                ])),
                Felt::new(BigInteger256([
                    0xc1fad51240166224,
                    0x429249dc25837edc,
                    0xd69ec9ba4fd243e1,
                    0x4a689eedc56609ff,
                ])),
                Felt::new(BigInteger256([
                    0x23fd32b962b15c37,
                    0x23f8873e0bf62c0c,
                    0xc832e064d6b3f22f,
                    0x312d528d4bb03620,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x505574f8c1ffc808,
                    0xbd7b233594c215ff,
                    0xd334ddacc26df14d,
                    0x5071c2016bad3953,
                ])),
                Felt::new(BigInteger256([
                    0xb64b1780e5156103,
                    0x8c0999a7e6fb0957,
                    0x3eba2e0fc999237b,
                    0x1eeb90536a2162f1,
                ])),
                Felt::new(BigInteger256([
                    0xc42da9d28c876109,
                    0x396fa91aae022042,
                    0x353ca2e97ceee0ab,
                    0x6b955323899b6964,
                ])),
                Felt::new(BigInteger256([
                    0x27e919b957867a8e,
                    0xeb28841253975d2f,
                    0xed1668bcc7e8b01e,
                    0x163ad2d427760ada,
                ])),
                Felt::new(BigInteger256([
                    0x1abedd56df5fd567,
                    0xc0fa2f5fdffee186,
                    0x17f3170529f8b694,
                    0x157991661bacbbb8,
                ])),
                Felt::new(BigInteger256([
                    0x2a0ee03a043a65ec,
                    0x02614e4d6b4c99bc,
                    0x9c78345735921a7e,
                    0x1e3904e9c4807486,
                ])),
                Felt::new(BigInteger256([
                    0xe1cc8d532d2c8931,
                    0x37d89b45d024e7b1,
                    0x0ecfe17c1d94c19b,
                    0x409dda17c4c15fa6,
                ])),
                Felt::new(BigInteger256([
                    0x3ea1c70af77eb052,
                    0x90b39012b0044b36,
                    0xa61807c14288aa4d,
                    0x050929b2779c682f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc255ed582121d436,
                    0x968d474cc5c08249,
                    0xb2021dad6e0a14f5,
                    0x5ad3f7c8ba50e64a,
                ])),
                Felt::new(BigInteger256([
                    0x2d9922660bb7cb3b,
                    0x797baa9f5b93ae8d,
                    0x4f1a7d873397615b,
                    0x47faa7c4b234c604,
                ])),
                Felt::new(BigInteger256([
                    0x57195dcd3985802b,
                    0x33d006925b9fc15c,
                    0x088154488eff9ba1,
                    0x20c163be04484720,
                ])),
                Felt::new(BigInteger256([
                    0x47a37db9eaabcfc5,
                    0x1978de770b1f932c,
                    0x53dbf19d631fa456,
                    0x03e661eacb6f17d0,
                ])),
                Felt::new(BigInteger256([
                    0xc4cf24de2961a9c3,
                    0x6f890a955b4c93b9,
                    0x237ddfe057f155bc,
                    0x647e46a0176b27aa,
                ])),
                Felt::new(BigInteger256([
                    0x5f5a9ee11f6cb6ff,
                    0x022344286d33e77d,
                    0x1af7c475574e03f2,
                    0x3502842cf3aaadf6,
                ])),
                Felt::new(BigInteger256([
                    0x25532f11841e4645,
                    0xafd5136771692983,
                    0x96f4f8b67678cbfa,
                    0x484bad1a0a35a074,
                ])),
                Felt::new(BigInteger256([
                    0xd9d8bd2c4f3064b5,
                    0x472f8df9c00bd4c7,
                    0x00d044b8a7c2ed2d,
                    0x1cee2ab1a6abb169,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x8141ad4098c12e04,
                    0xdf2dfdfb3bd02d29,
                    0x0549b05bd01d5cc9,
                    0x4d453abd45dfa905,
                ])),
                Felt::new(BigInteger256([
                    0x4cdb80ab97f33302,
                    0xd9b29611ce7129e4,
                    0xb5e0804815e94ae7,
                    0x5395734a6a8dbebe,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x29c951f9d2808e73,
                    0x0c4848e37d946028,
                    0x90a06d348cf04a6d,
                    0x057c0d18587b1dfa,
                ])),
                Felt::new(BigInteger256([
                    0xf3b1c59d05a9afba,
                    0xec8f725d9f6d60fd,
                    0x873de0ab2b02f88a,
                    0x1334a4f7de90fb98,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xefd65ce5512d818e,
                    0x857ddd16e6a103f8,
                    0x586dbecfbaa73bb3,
                    0x593f8c9b8da19f22,
                ])),
                Felt::new(BigInteger256([
                    0x2dd92c7761c65f48,
                    0xb34e8c382de2e6df,
                    0x0529d68789a90109,
                    0x2eedf0c3ac3a9d82,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x31b2f62ed0e75a89,
                    0x00541d6c803c248c,
                    0x2fd23469a5d9889f,
                    0x50350a21303bb2fe,
                ])),
                Felt::new(BigInteger256([
                    0xd7ce7fa4880d257a,
                    0xbdd0718fe60e5268,
                    0x8e01c0963f2d2b44,
                    0x57fe2f8d3d0af02d,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x941942b174e1e9b9,
                    0x3e670f53263e0709,
                    0xe9967644cbd2a23d,
                    0x2a9fe0b2ebb2f48e,
                ])),
                Felt::new(BigInteger256([
                    0x5e174c623a8aa25c,
                    0xc8deed934f7cbe36,
                    0x5594fb1feb9a15bd,
                    0x3098cc81e771531c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4144026722b07960,
                    0x99e82a478d7cb04d,
                    0xd58c317accf2ea97,
                    0x55367bc740bf61a5,
                ])),
                Felt::new(BigInteger256([
                    0x60ae958079c436a0,
                    0x43b0e7288ac6e39f,
                    0x9a9c9664272765ed,
                    0x35435b743bf66eca,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x35ff525345b00a9a,
                    0xe6706e09d2d400b0,
                    0xbcdcb360c3c95555,
                    0x624e886677b50ace,
                ])),
                Felt::new(BigInteger256([
                    0x24c945f982839bb0,
                    0x7e6a31fe751cfd30,
                    0x8840de0b6716f950,
                    0x5ab75e3c7b2900d3,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf40d85615ca684ad,
                    0x36760c99f94ce0c9,
                    0x8cd2a0525ce31f9e,
                    0x594b4f207fb55e0a,
                ])),
                Felt::new(BigInteger256([
                    0x8535f42621145fd6,
                    0x08eed17b727368ea,
                    0xb5da9d17e17506a4,
                    0x4a65ae37686d4441,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe0742e48525b7856,
                    0xe8a1aef193e20ade,
                    0xa90d160827f2917f,
                    0x4f66986849ae6228,
                ])),
                Felt::new(BigInteger256([
                    0x6a3a026a7da084a5,
                    0xd7fcca326cc3ef0c,
                    0xb613cbf1202039a9,
                    0x724eb76be879f301,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xeb979841644a967b,
                    0xa11b235fcc633bfa,
                    0x694a8444983db306,
                    0x33427f0104c4b936,
                ])),
                Felt::new(BigInteger256([
                    0x9fe3e7bc721a0c68,
                    0xc15d701b1cc54c5a,
                    0x393c91bcc8077fd5,
                    0x4b2beb4b147efee8,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
