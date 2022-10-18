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
        assert!(k % 2 == 0);

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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0xa597051421fbac91,
                0x3f6894a88526da39,
                0x7b3e50e1ac0cfcf9,
                0x238180c5e352d68d,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x76dafe854154d18e,
                    0xb390289a118d48d3,
                    0x7f1a139cd3280486,
                    0x4621151b74f5c984,
                ])),
                Felt::new(BigInteger256([
                    0x8785218d37b8f1b2,
                    0xdb600787995ac12a,
                    0x1643bad26dc4090e,
                    0x63387769a442e76f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb0f28f5727db1478,
                    0x7bea82d5e196d48a,
                    0x096deae7175f039e,
                    0x37574a21c31b5466,
                ])),
                Felt::new(BigInteger256([
                    0xaa370b1dedbe56ff,
                    0x0f925af52bfb2282,
                    0x1db7111a02aab299,
                    0x5631db6d70ff9f57,
                ])),
                Felt::new(BigInteger256([
                    0x1fdaffe2fa8361be,
                    0xe41a73f98e8df430,
                    0x20f9a80cf4c23a79,
                    0x45e13d67223db77a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x91e7cfff9e2dda50,
                    0x84fc0b27726d59f6,
                    0x26d00565f196bce0,
                    0x282f9f679553e014,
                ])),
                Felt::new(BigInteger256([
                    0xa23a1561e2959aec,
                    0x5b0b61ae4ae7f18c,
                    0x0d2eb05bf430c13d,
                    0x518e315707a6e969,
                ])),
                Felt::new(BigInteger256([
                    0x9004bdb25d4a6acb,
                    0x23d6aa25156114ce,
                    0xb354cf6521ad8949,
                    0x33aebf30acb3e1f8,
                ])),
                Felt::new(BigInteger256([
                    0xc8451f281cd8d0ee,
                    0x76e843808b206b74,
                    0x96a3c0fc188e3fc0,
                    0x2a7b412b4d7e1d29,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xaefad14ad904d698,
                    0x5a1be95df620e646,
                    0x4592d97c7cb78292,
                    0x6267acd3a895aa56,
                ])),
                Felt::new(BigInteger256([
                    0x07451d69fbc4930d,
                    0x3d93ca9f34d99492,
                    0xfefdaca3567386c6,
                    0x4387d9b1b0e05f6d,
                ])),
                Felt::new(BigInteger256([
                    0x883998d5a927a5fa,
                    0xb2f192c85a944b7c,
                    0x12d21535555e0930,
                    0x22facbacc3c3f79f,
                ])),
                Felt::new(BigInteger256([
                    0x08e72ac16092adb5,
                    0x6d55945f708828a0,
                    0x28060c3ca4e2bfab,
                    0x095d9b0bf7983515,
                ])),
                Felt::new(BigInteger256([
                    0x9d583d0f9067a8c2,
                    0xe518340e9c5d53e8,
                    0x9e4e65250942e833,
                    0x22c691a5809fcd7e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8a84cbb415af4353,
                    0x5f354ff9ee5648ae,
                    0x6327a84bded6506f,
                    0x681d46259edd141a,
                ])),
                Felt::new(BigInteger256([
                    0x7cd96a12bcd92a0b,
                    0xc7f2ca3f88a3b86d,
                    0x3953528e60d028aa,
                    0x63aea627219c3fa4,
                ])),
                Felt::new(BigInteger256([
                    0x6bfcab361a78fe6d,
                    0x133f710e1bc4c8f0,
                    0xba9b0c76bb8d55a4,
                    0x1871356f84703eff,
                ])),
                Felt::new(BigInteger256([
                    0x0fd44ab2826595a7,
                    0x9d8d08a50c9f2756,
                    0x545d60b01722989e,
                    0x2eda37dee56a99dd,
                ])),
                Felt::new(BigInteger256([
                    0xa33a50cc075f20d5,
                    0x5edb9aa51cbc9446,
                    0xd2693ee6cf3e7c38,
                    0x3f130d3fd07bbcaa,
                ])),
                Felt::new(BigInteger256([
                    0x04b7e9eb6a7474b5,
                    0xed33b6ed90271737,
                    0x0e88d590884359ed,
                    0x1e411953a56bb0f3,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x484035308bbd3aca,
                0x7346799a9247c6a9,
                0x8d8e6fc582320e49,
                0x58c7b8974e67ed37,
            ]))],
            [Felt::new(BigInteger256([
                0xd1ea5feb165de7ce,
                0xd52f40811e521ec1,
                0x0c8635510415ad4f,
                0x3658ae6c085ce085,
            ]))],
            [Felt::new(BigInteger256([
                0x3f9a3a60367fb2f0,
                0xeee0400ad28f2d73,
                0xb54bd7140e7558b3,
                0x0ae3df469c126c91,
            ]))],
            [Felt::new(BigInteger256([
                0x38f5ab96944f8898,
                0xbe0073f9318f7ccb,
                0x2ea82fa4db1e553c,
                0x00d2633bfbe100a8,
            ]))],
            [Felt::new(BigInteger256([
                0xf5815e6310743fa9,
                0x5d24fb5784b1e76b,
                0x3b78f60a99dd45ed,
                0x239cae14d15b50a0,
            ]))],
            [Felt::new(BigInteger256([
                0x43691f8286f7d97c,
                0x589dcdce87445d64,
                0xad8a5c1ad1463f69,
                0x5f8ce70551310021,
            ]))],
            [Felt::new(BigInteger256([
                0xb8a01162d0674dd4,
                0x4d5d525c827da38e,
                0xc86d76f4f46b8eb0,
                0x616018cd152f47b8,
            ]))],
            [Felt::new(BigInteger256([
                0x693296156fd3dde0,
                0x44d32dc045e5bc62,
                0x80256264ff64c68a,
                0x1e9032dc26b8eee0,
            ]))],
            [Felt::new(BigInteger256([
                0xad7f78f3f18aaadf,
                0xc4f21d3b5910c483,
                0xa17d0fa6ad979528,
                0x61ed52d4bb905111,
            ]))],
            [Felt::new(BigInteger256([
                0x75943d846b8c6ae7,
                0x64688eb4784e81ee,
                0x46947dc9dbb11022,
                0x32ee09c4b3ab203d,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x5e6fce0149ea1e23,
                    0x7c5f436fefa51834,
                    0x56f41d326c84f2c1,
                    0x3ef48a9388373cbc,
                ])),
                Felt::new(BigInteger256([
                    0xcfe7f025aca47a7b,
                    0x0556e34314bd32ae,
                    0xb48b7488967c13d8,
                    0x6006ca11e11ee4af,
                ])),
                Felt::new(BigInteger256([
                    0x589d660d6ffb731e,
                    0x7c2a1aa90777eeb8,
                    0x6f72a8aa03f6415b,
                    0x29490fc4ef27bf6b,
                ])),
                Felt::new(BigInteger256([
                    0xe600844a00861fae,
                    0xbfb1933aefa57879,
                    0x9b368b4f7f5dc231,
                    0x62946bf3981e1950,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2ef38482a2c131c2,
                    0x662497b3ee09535f,
                    0xdb544766281878fd,
                    0x54a9c81cc7d79cc8,
                ])),
                Felt::new(BigInteger256([
                    0x2b9cccbfa5160783,
                    0xe9a4e629b30561b0,
                    0x76e7a83d0ecbed4f,
                    0x0396c73de8f81ed3,
                ])),
                Felt::new(BigInteger256([
                    0xafa99fa855c9cfbc,
                    0xca8d69da158506e7,
                    0x13e098f85640aeaa,
                    0x1f1b358e4497042d,
                ])),
                Felt::new(BigInteger256([
                    0xd0905233d23ea38a,
                    0x21b034d54ff04afd,
                    0x24141d911250c91b,
                    0x2f46af9378dbb441,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x29b578237bb119e4,
                    0xa22b1991c21d750a,
                    0xe297f365868e98e6,
                    0x235d05d771e1f1f9,
                ])),
                Felt::new(BigInteger256([
                    0xdccd52fbca8c4d5d,
                    0x8794bf8c16cfc5e0,
                    0xac94f67aa3adb83d,
                    0x3af7087c9db6da3a,
                ])),
                Felt::new(BigInteger256([
                    0xc34c6343d702cad6,
                    0x5a5a918c45098e71,
                    0xa8642fd50d5b3f41,
                    0x0eb296591a67ac21,
                ])),
                Felt::new(BigInteger256([
                    0x5b3a2c8741dde2b9,
                    0x429e02da739d2a51,
                    0xa0eafd809a7de48b,
                    0x24cc50857bc7a5af,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9ae3ef8d79d50eac,
                    0x26a4149ad2ad9b26,
                    0x3ce8e6dac5aeb0b8,
                    0x6c3ed2f18afd3d1e,
                ])),
                Felt::new(BigInteger256([
                    0x31032c1d43bf2d1a,
                    0x597179631469afdb,
                    0xfcad677d7db720dd,
                    0x4bd495faff697d0e,
                ])),
                Felt::new(BigInteger256([
                    0x5810320145361454,
                    0x8fcf7630a6a6919a,
                    0x0951e50a4e8d4e51,
                    0x6a045ccd88b193ba,
                ])),
                Felt::new(BigInteger256([
                    0x1199691f8ce84273,
                    0xf99a5805f588148a,
                    0x7411ad134e4b624a,
                    0x24fc962728d580ed,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x901d400624603308,
                    0x3bcc5a59b2ec40b8,
                    0xa6d6c6a9bff656f8,
                    0x1896dcf97500e69a,
                ])),
                Felt::new(BigInteger256([
                    0x6455594aaed6bed1,
                    0x6eeae2a528648f3e,
                    0x6066bd6b5b2bdb5f,
                    0x577e743d2438214c,
                ])),
                Felt::new(BigInteger256([
                    0x70c12f6c3773321b,
                    0x0c1cfee030252984,
                    0x473b3f933f8189a2,
                    0x21e2f1d9df811901,
                ])),
                Felt::new(BigInteger256([
                    0x10fdbb99e2a4b348,
                    0x124f2efecb94d33c,
                    0xc428305aad50b157,
                    0x342c86edc0a025d6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x5e2d81eea0c83a5d,
                    0xb6ee44f41a8f4323,
                    0x408d1ab5d9d5c621,
                    0x21d9ea9a4747766a,
                ])),
                Felt::new(BigInteger256([
                    0x4ace2929218b0290,
                    0x9caaefe81a9fe380,
                    0x28798726e5a9ae9a,
                    0x0545531d9ea315ed,
                ])),
                Felt::new(BigInteger256([
                    0x812dfbfb2fee38af,
                    0x408e29d8fceaa09c,
                    0xc8f78e6fb7f38d5f,
                    0x1d31c37c1c165d67,
                ])),
                Felt::new(BigInteger256([
                    0x298346001900ef1d,
                    0x837737f35318e5bc,
                    0x65c7c941b608e4d1,
                    0x1e86f577f8d4390f,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xe01205e97938396f,
                    0x3c5f0a0c772ca5fe,
                    0x4d7c683d5be812b9,
                    0x0ef81890105ab0e5,
                ])),
                Felt::new(BigInteger256([
                    0xbe3d2210fe19cad4,
                    0x84965e5a2026b6f5,
                    0x0da8400dacfc2d9c,
                    0x0aa6b3cc492f8646,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd0dd0a3d01672917,
                    0x82d109118f47bc97,
                    0x47609de8b4ee8598,
                    0x08080c4aa288598d,
                ])),
                Felt::new(BigInteger256([
                    0x96fcfb100b38743c,
                    0x2ec7dd9b27714296,
                    0xdbd018ea8b479c77,
                    0x62fd401b7ec38d95,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe9be2ac6849d2a9c,
                    0xd2d4d214e5556401,
                    0x6c6dfdd4d14011ff,
                    0x6c067d072158ed91,
                ])),
                Felt::new(BigInteger256([
                    0x35461c3a40d6277d,
                    0xb451e1a27a524470,
                    0xed04e1f1ecdb1371,
                    0x56193d444364ba9a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4e1ea29a091f62eb,
                    0xc1ec8d84059608ad,
                    0xc4c5945ef8d2e4bb,
                    0x3800a86659000630,
                ])),
                Felt::new(BigInteger256([
                    0xecf26b6c8d8141f8,
                    0xc45db77374d8d741,
                    0xac69914e2747af72,
                    0x3138df4d1f4e2adb,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0c02363d775a89d0,
                    0x782acc3d67d732cb,
                    0x7caddcfd3cc62d08,
                    0x1153d270a0daaf39,
                ])),
                Felt::new(BigInteger256([
                    0x36f2f8fd1cd74b64,
                    0x152f4a1957d7a791,
                    0xf15323c6e3f4a644,
                    0x6e03a87070671075,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0e208cbd872637ce,
                    0xb59862958634e49d,
                    0x1fd055b8f9f4c8bb,
                    0x232cc78995352899,
                ])),
                Felt::new(BigInteger256([
                    0xefb9e2c61c53af4a,
                    0x64b6a3a505232a9a,
                    0xc146b0be6236e707,
                    0x60bf67ae58ab4f9a,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xebecd48322ea6f64,
                    0xd7e2d78ff67e5e9b,
                    0x664c53f33dfafdae,
                    0x6d938cb219428d17,
                ])),
                Felt::new(BigInteger256([
                    0xfcf3c13e3204f8f8,
                    0x9d1e2ad04f6fee61,
                    0x6306c2812dc52fd9,
                    0x113ec6b7fb38aa42,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6b2b19128400b06e,
                    0x3300e2c2059eefe0,
                    0x58655abd9d15cd94,
                    0x6eda356d9dc21989,
                ])),
                Felt::new(BigInteger256([
                    0x9ecba100e6c07681,
                    0x0c87f789a4d21a7d,
                    0x2b284ad382d004d5,
                    0x67096838c8cd292e,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0e0d2bb703249922,
                    0xac1177a74d251089,
                    0x702eaddd1268d7ff,
                    0x116afb1d69f425a0,
                ])),
                Felt::new(BigInteger256([
                    0x295b5e4f2ea71f15,
                    0x971b408a7e8909f0,
                    0xebc1e8d209cee80a,
                    0x0daa2db4b6313020,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd2d98fbbe3bdc851,
                    0x85745a1ff49ebfd6,
                    0x60c4f2adab3cc167,
                    0x2c9c7d9e8e7abcbf,
                ])),
                Felt::new(BigInteger256([
                    0x9970bb793cd56855,
                    0x0d17ca2977c5f2d2,
                    0x3f18d083d75e16ef,
                    0x497a148bc92bf085,
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
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x2acd9a92550375d3,
                    0x3777c0bf93130f71,
                    0x7533d86396be7725,
                    0x37459b00e621920d,
                ])),
                Felt::new(BigInteger256([
                    0x7e9ecc0cda078adf,
                    0x30571d7809334175,
                    0xec8611470458a2e8,
                    0x08b046f53028efc3,
                ])),
                Felt::new(BigInteger256([
                    0xbaef1d2f11d46b48,
                    0x64a924a7ab0c6010,
                    0x133123a66c132fe6,
                    0x4b37ef2032c94481,
                ])),
                Felt::new(BigInteger256([
                    0x5c4b1fe353d1c17c,
                    0x63c54e69e51dbaa7,
                    0x056a167eb6bf9051,
                    0x6784a4e5f1744d29,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0b13144a47dcc65b,
                    0x1366d63c878518c6,
                    0x07ef1ac0aab8d9b0,
                    0x0c0cfb544aa056b1,
                ])),
                Felt::new(BigInteger256([
                    0x22d32e0a168fee40,
                    0xa0c58616e47c8c28,
                    0x5b144a2ab79267da,
                    0x13620783704f4e8b,
                ])),
                Felt::new(BigInteger256([
                    0x23bb96e14b7753d1,
                    0x33fdd34d268dd225,
                    0x54d9750a347e7142,
                    0x6173c9ba30db03dc,
                ])),
                Felt::new(BigInteger256([
                    0xf3d1cc8d46c35ef1,
                    0xf9244bb3d6b334d6,
                    0x40e1a04bebe83bac,
                    0x18ab6a2d913bf0ea,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x087c84e42c9ab28c,
                    0xbdd765c1f84efa9a,
                    0x4d2051a212265f87,
                    0x0ac3b4a562bf18c9,
                ])),
                Felt::new(BigInteger256([
                    0xd053b8a2334a12a7,
                    0xc5d4913faa3fa7ec,
                    0x21477756515e6860,
                    0x22c558d1d5ad6cb5,
                ])),
                Felt::new(BigInteger256([
                    0x8a19f465c022536d,
                    0x05890aa28e8db57f,
                    0x96bb76877e7af790,
                    0x47a36ee8d8db933c,
                ])),
                Felt::new(BigInteger256([
                    0x8d9a524ba92123ac,
                    0xd601a271d40b57a5,
                    0x31442da4f92ef035,
                    0x0ae41a539f7a8f11,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1ea09d14008fde70,
                    0xe4d6443c6b85e084,
                    0x50aaaa7ae5e9baa5,
                    0x5db5b2444c144244,
                ])),
                Felt::new(BigInteger256([
                    0x27c1ae625eb38b1a,
                    0xe906836d8a484ebc,
                    0x866324b32b2ebc35,
                    0x1ea18a5ec8cd6784,
                ])),
                Felt::new(BigInteger256([
                    0x0d5d45da25c10d87,
                    0xc46bd95661da2edc,
                    0x58d81570b36cda0b,
                    0x41d82a9736c99acb,
                ])),
                Felt::new(BigInteger256([
                    0x03c9c110338d5efc,
                    0xbe4a383d48d4342f,
                    0xb3cf516e841cb2d2,
                    0x1e4c0b5184d7024d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbc4dc42e6fd4d646,
                    0xd798d4f1ffb15390,
                    0xe2ca24eeb7ad6817,
                    0x1e4d824a9932eb0b,
                ])),
                Felt::new(BigInteger256([
                    0x323ac52e40a9c7f7,
                    0x79f5dbc624bcf96e,
                    0xb19e776ed2c2d907,
                    0x2e6dade76cd4fd9e,
                ])),
                Felt::new(BigInteger256([
                    0x14c28847d2f66c14,
                    0x028383fcf3355060,
                    0xe9b6d750a52d7625,
                    0x5b405d99530bd8b4,
                ])),
                Felt::new(BigInteger256([
                    0xc9530352c119e8e0,
                    0x9b6f9234379f008e,
                    0x1d3c98cf23a90d62,
                    0x5631e3e249c0c182,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdf966a2cff350393,
                    0xd9171ccc820fdfc4,
                    0x47a500f06d764f4b,
                    0x4d2855bb43e01582,
                ])),
                Felt::new(BigInteger256([
                    0x63512498ed8dabb6,
                    0xcdbbd8d64efbded1,
                    0x901672f936728ece,
                    0x2998e3c5c0871c06,
                ])),
                Felt::new(BigInteger256([
                    0x71db111b8fb82e0c,
                    0xe7c90980dd677f32,
                    0xb5f2c393596d5908,
                    0x11d7e5d9c4c5419a,
                ])),
                Felt::new(BigInteger256([
                    0x91a2564bbf9792e9,
                    0xa12554976c543a30,
                    0xff9222af236c4dd3,
                    0x29a94bf95001b926,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x9e4f27fa77520443,
                0xc0f5686697535cf4,
                0x5b24a84b08e44055,
                0x199ecc5c598a372b,
            ]))],
            [Felt::new(BigInteger256([
                0x67da054d0c9f9d53,
                0xb198e6acb6b8ff2e,
                0x2330b6d34036220f,
                0x6b054c66214be723,
            ]))],
            [Felt::new(BigInteger256([
                0x1f044701c5735218,
                0x33690fb45fa94c73,
                0x263907beb4794d6c,
                0x4e3212f83b202ae4,
            ]))],
            [Felt::new(BigInteger256([
                0x3b110e0696a0a4e3,
                0x864a44f77a6edfef,
                0x712f25ad201a942e,
                0x693987b3784e310c,
            ]))],
            [Felt::new(BigInteger256([
                0x609875aa6712236d,
                0x84e891690fc73830,
                0x9cdbe7bd08c4f032,
                0x080348311829b98a,
            ]))],
            [Felt::new(BigInteger256([
                0x33fcb7315cd501f5,
                0x56bcedcd33ff6ce0,
                0x69cf69a4bbe35ea3,
                0x445f840d35108f71,
            ]))],
            [Felt::new(BigInteger256([
                0xd9071716fa954913,
                0xcf53edf21a261c85,
                0x46055b261e90a4d6,
                0x25975860c5faab3c,
            ]))],
            [Felt::new(BigInteger256([
                0xfc1cd5463a15c7ca,
                0x2fba413c480db17b,
                0x36d67a1b0e2f4178,
                0x1bf225c077b381ba,
            ]))],
            [Felt::new(BigInteger256([
                0x5663c9c81723b107,
                0x5d289793d329cb8b,
                0x2290a2266088a6c1,
                0x0d4e0404d52cfc6e,
            ]))],
            [Felt::new(BigInteger256([
                0x0d4c6c46f99f8a91,
                0x7a57267158110ef6,
                0x24b44cbd82155d1a,
                0x147a7145442dabe5,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 4));
        }
    }
}
