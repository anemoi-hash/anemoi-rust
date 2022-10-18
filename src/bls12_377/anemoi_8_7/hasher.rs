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
        let num_elements = if bytes.len() % 47 == 0 {
            bytes.len() / 47
        } else {
            bytes.len() / 47 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 47-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 48];
        for chunk in bytes.chunks(47) {
            if num_hashed + i < num_elements - 1 {
                buf[..47].copy_from_slice(chunk);
            } else {
                // If we are dealing with the last chunk, it may be smaller than 47 bytes long, so
                // we need to handle it slightly differently. We also append a byte set to 1 to the
                // end of the string if needed. This pads the string in such a way that adding
                // trailing zeros results in a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 48];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 47 {
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
    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
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
            vec![Felt::new(BigInteger384([
                0x16712fa98c1212a8,
                0x3f2ce7888759a0d5,
                0x1dea92e3981e4162,
                0xddcb76107fc09f9c,
                0x7a3dd1c7a04b0f56,
                0x01a30b7e42888484,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0xca03d461588b1453,
                    0x13938393d4d859da,
                    0xe22efc714ab84ea5,
                    0xcd31fe25a20d0a1f,
                    0x0695e1d78e9c28c3,
                    0x013e673036209668,
                ])),
                Felt::new(BigInteger384([
                    0x41a62f4ee802086f,
                    0xb451c85a1f0ad2ae,
                    0x78a8aa5fa6b7dc98,
                    0xb4ed61a5f290f60c,
                    0x98420ac8c83f8709,
                    0x004938c26c8f08f4,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x84775c2c8c4d1534,
                    0x311caacc04f07070,
                    0x2df8823ad910fa42,
                    0x772166ba877b6edd,
                    0x808725086b52356d,
                    0x0182da8f3f47a1b1,
                ])),
                Felt::new(BigInteger384([
                    0x0a2706c89e699ce2,
                    0x9c100c6b5adb3729,
                    0x840a282ef93ed5f6,
                    0xd86208e0e20d3800,
                    0xcf848cbcb0719db5,
                    0x01911f85fdcc756b,
                ])),
                Felt::new(BigInteger384([
                    0x37a041570966089a,
                    0x0ba5c06d8951bb15,
                    0xa7b8a212b4e71441,
                    0xc639b231112d3ba9,
                    0x9bc9448b137f25fd,
                    0x00a69b866f47ae97,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x983c23e00042512f,
                    0xe89b4ee2959b71b0,
                    0xdb47609120fade68,
                    0x27afff0fcb7d06ef,
                    0x25dde32e9c6e22b2,
                    0x0160edd0a6e436ff,
                ])),
                Felt::new(BigInteger384([
                    0x1f0b4c398303c93e,
                    0xc039b79a91e97288,
                    0x17f3bca887ef63a1,
                    0x8b1bd217eff74188,
                    0x25e458847e9949c4,
                    0x01679d759a88a79d,
                ])),
                Felt::new(BigInteger384([
                    0xae95879cb32ef3ed,
                    0x24650e10708f2978,
                    0x800abcb05106c0ec,
                    0xab05c0133ae5cb9b,
                    0x3b83d83598cd215f,
                    0x000522ca2f1601a0,
                ])),
                Felt::new(BigInteger384([
                    0xf41a2edd5801f123,
                    0x58e4c6d3bd18ae8f,
                    0xf8154388696c3e54,
                    0x3ef4dc6ab27c30f0,
                    0xafdd4ababc3b32fb,
                    0x013f3c541f8c4681,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x1c7dde9e2114e27f,
                    0x604bfc500aadf2ca,
                    0xc062a68b69e0d0f7,
                    0x878f86b9f180d85e,
                    0x8871f5f2d0ad8fa9,
                    0x00ffeade2be86938,
                ])),
                Felt::new(BigInteger384([
                    0xbfcd223f6ea3229d,
                    0x3840af03ee5a1458,
                    0xace0de47834d126a,
                    0x4ee5e80c8172a97e,
                    0xfca9e2af6a15a71e,
                    0x00ab6a039ed01a7a,
                ])),
                Felt::new(BigInteger384([
                    0x4283789d732ae0d2,
                    0x3e09f9f4e4f2942e,
                    0x0c46da9416f14aac,
                    0x9bbd10459fb41153,
                    0x38c8695ec851fe4e,
                    0x0020e0596e9fa08d,
                ])),
                Felt::new(BigInteger384([
                    0xa1b48cb1cba510fc,
                    0x8b07aaf5b81c72e1,
                    0x1537a6b261a26fe0,
                    0xebae1a3cad28a747,
                    0x89cb56f3a48a2fb7,
                    0x0151b92e91495263,
                ])),
                Felt::new(BigInteger384([
                    0xe07fda2d8bbd32ef,
                    0xb23696a43eafd272,
                    0x18b0bcd9a2113e29,
                    0x552131def2d3f95d,
                    0x5b0032726321d9a3,
                    0x00222a78902902f6,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x22ae53ac05ca087e,
                    0x0ead9a3544fbc60a,
                    0xbae30ae2ce12b9eb,
                    0xd15fb7961ae3827d,
                    0x72b7ff317796eda7,
                    0x00e1addcef73486b,
                ])),
                Felt::new(BigInteger384([
                    0x4b533302b3e0fcf9,
                    0x0937824ef0e6c346,
                    0x093360054c9d089a,
                    0xccaddc3a7f8f8523,
                    0x1022011179bb9838,
                    0x010d0c02fb5163b3,
                ])),
                Felt::new(BigInteger384([
                    0x4ad1791a6d6e7520,
                    0x25ba37fd1eb79da0,
                    0xdf31c9f607ca0bd7,
                    0xff1947dacf57e77f,
                    0x96d1ab86fb40a8a3,
                    0x00dea8c3ada327cc,
                ])),
                Felt::new(BigInteger384([
                    0xafc2e42abe613e19,
                    0x781b7984f273061a,
                    0x8fbf62e5ccf4cb4a,
                    0x8ac00e56be10c743,
                    0xcd99efca2ec4b555,
                    0x007567c53f82d9a7,
                ])),
                Felt::new(BigInteger384([
                    0x905faa9ec04bd16e,
                    0x95156a4e07978937,
                    0x361fb3aacd688fc3,
                    0x6d2d0806112ebf00,
                    0x7b55b09ab7403ccd,
                    0x0069d3dc6860f96f,
                ])),
                Felt::new(BigInteger384([
                    0x47cea6856bd3cd4b,
                    0x5e8f2312b8d3eb57,
                    0xb7fd9ea16a000bdb,
                    0xa959d61468bedc3c,
                    0x31ebc830634ab85e,
                    0x00139f3e4da25fb9,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x831c362d17044453,
                0xd34ee4464e6db24a,
                0x8a6e9e9b29b97d1d,
                0x4cbc52b4630dbee9,
                0xf0ea5f5e64b71868,
                0x0048fd2993eb4b67,
            ]))],
            [Felt::new(BigInteger384([
                0xfa82c9e27dd1b70d,
                0x650e7527a9ca89ba,
                0x4998e9fbb4b68331,
                0x4bbc6ce8ad8c5e2b,
                0xe4156824164c53eb,
                0x01abeeb55042f4bb,
            ]))],
            [Felt::new(BigInteger384([
                0x84a53172abf6a1f6,
                0x6f6ca39b8afe567a,
                0x29cdb9c49455d5d7,
                0x87834bd7b51ca535,
                0xb61ed95fed06fffc,
                0x00eeeecc1d7a9cad,
            ]))],
            [Felt::new(BigInteger384([
                0x280324b32fa62c1c,
                0xfb1b52fdb01de783,
                0x58f58f58703cf844,
                0x8675d7090e5c92b5,
                0xeb2fd3020efdb61e,
                0x004da6d976ad0b68,
            ]))],
            [Felt::new(BigInteger384([
                0x505bb4b35aa81910,
                0x96f9e769b84d069e,
                0xa80716406411ffd7,
                0xfbcef4fee2e9f3dc,
                0x7543170d8371c66d,
                0x01541cc9f0093abf,
            ]))],
            [Felt::new(BigInteger384([
                0x35d78cbddcdde6b6,
                0x75989a3ff2b572d2,
                0x8026d260d7d6a0ab,
                0x8a91bb7f8ba215ce,
                0x7fd0d565e36ed68e,
                0x00e1b7820e57e3df,
            ]))],
            [Felt::new(BigInteger384([
                0x39ba76fa6db3e42a,
                0xc3807ad0d3987fb7,
                0xc16bda97fc649caf,
                0x576d89880b4e5f0c,
                0xb467834666afce50,
                0x002e09802beb0927,
            ]))],
            [Felt::new(BigInteger384([
                0x543de8cec9317076,
                0x19b3b4d29047fa56,
                0x8e0c01c4a595f300,
                0xbcc76992211f7e65,
                0x2f2d6343f50e7a90,
                0x019950ec1169820b,
            ]))],
            [Felt::new(BigInteger384([
                0xeab48ed0651f5aaa,
                0x6c58d4901d3db67d,
                0xf2dd9db909a79134,
                0x8b85887e30cd0115,
                0x7497db0eb55817ca,
                0x006cc17405232f3a,
            ]))],
            [Felt::new(BigInteger384([
                0x989b4efc346efb7a,
                0x8419ca03709e37cc,
                0x33043c3000ccab3e,
                0x01b04b3f156ea846,
                0xc62bc1d15dcdb74d,
                0x006e405dd83fc19e,
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
            vec![Felt::zero(); 8],
            vec![Felt::one(); 8],
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
                Felt::new(BigInteger384([
                    0xfb57d53f7474649e,
                    0xd0f428022cd5c614,
                    0x74e1ab7ffc98813c,
                    0x5efbee18331fa2aa,
                    0x4e03f9673a3b52ca,
                    0x0198da241410ce64,
                ])),
                Felt::new(BigInteger384([
                    0x0e50d8a77744f9ad,
                    0x5bf51cf733e20206,
                    0x6f067c450f0bd62c,
                    0xe96399db7eca3474,
                    0x1ba7d257f60e46c5,
                    0x0112bdd282e2e107,
                ])),
                Felt::new(BigInteger384([
                    0xaf54d7a5634a4f80,
                    0x18f3e30d9a7f863f,
                    0xa554a28df43e13ae,
                    0x0fa9a983d57e7927,
                    0x03fbdfc1f2e2f664,
                    0x00f140275f5879f6,
                ])),
                Felt::new(BigInteger384([
                    0xcbc12e912fa033cc,
                    0xf138e97777e22c24,
                    0x7381c0060376f419,
                    0x5e22162408562859,
                    0x573e5c1816538b0c,
                    0x0137ed8192a316bf,
                ])),
                Felt::new(BigInteger384([
                    0x3c8dbe1f6f6e21ab,
                    0xb692ecffedd5856d,
                    0x45c3f93698ee8719,
                    0xdd3d463cc22b458e,
                    0x95ae48da81d1a43e,
                    0x000d143d9d3421af,
                ])),
                Felt::new(BigInteger384([
                    0xdd6df4312db86367,
                    0x0651e9927e2fa87b,
                    0x10952d82334d9b18,
                    0x491b776d3c4a6979,
                    0x82a2badb3c7a89d0,
                    0x0104ccb16450d338,
                ])),
                Felt::new(BigInteger384([
                    0x24f0f40ebc8b8160,
                    0x37246b90a55bdbfc,
                    0xeb73c81622537e06,
                    0xa89ff293c3a2ba2d,
                    0x482ea4541d46756f,
                    0x015f60ddf185da0d,
                ])),
                Felt::new(BigInteger384([
                    0xdb81743e664fe410,
                    0xaebea624a02fbfcc,
                    0x5f7a6013e21477b5,
                    0xee0236e077a8ccad,
                    0x285033cb449c9085,
                    0x008aceca0ad072a9,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe949e8b16846c2f4,
                    0xe4ff0e6aad90ecd0,
                    0x4dbbb9c52dc4135d,
                    0xcedda1b7d6a4df37,
                    0xb996a9df05b7626c,
                    0x014c274ffdce5db8,
                ])),
                Felt::new(BigInteger384([
                    0x2ed1fbaaf56227dc,
                    0x5402558f07c7209a,
                    0x911a717b20d77fa5,
                    0xffeab38fb95168d9,
                    0x39f24047a26b69e4,
                    0x007e5a35249e8d0c,
                ])),
                Felt::new(BigInteger384([
                    0x54371e93454c3015,
                    0x0e05b8a28a333952,
                    0x784dfac802cd32c2,
                    0x2d87d5d8393d4895,
                    0xd940c27d00e108b7,
                    0x0124de590a8b4e03,
                ])),
                Felt::new(BigInteger384([
                    0x9363e86d9420885a,
                    0x28da315eafc34071,
                    0xc2d05b466bc39052,
                    0x3f8663ff223472c3,
                    0xc7dbd060cb5b36b8,
                    0x00af7fd93f4cf85e,
                ])),
                Felt::new(BigInteger384([
                    0xe5e13d619106e993,
                    0xdd730aeb1802ca34,
                    0x7faff6db073777e6,
                    0x46ce724ff28cda28,
                    0x175ae906420cbe3b,
                    0x01aa16c483435926,
                ])),
                Felt::new(BigInteger384([
                    0xba279e5212503fbf,
                    0x7898cb8c7bd6e55a,
                    0x53bb1dc8b3612566,
                    0x53e61ff9eea7a530,
                    0x386fd21cdb99c195,
                    0x01718a55ab08b542,
                ])),
                Felt::new(BigInteger384([
                    0xce604cb26da12c64,
                    0xc4f07fb7bd430a80,
                    0x78d0389fbd1ffb5d,
                    0x16997afa0e50b5d8,
                    0x3440fdca1aee9b2e,
                    0x00530992b6025cc3,
                ])),
                Felt::new(BigInteger384([
                    0xd84c1e5e3742cbfd,
                    0xf9679a5b09bf6d1d,
                    0xf2622e3100c8a0a5,
                    0xb21f174d8cc3557f,
                    0x28d98f6893562606,
                    0x003311a7a7614504,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x273129b2556103da,
                    0x47a363c58a5749fd,
                    0xb0d79bd7ad3dc630,
                    0x2a4125b42dc25d6d,
                    0x7694b8963515e21b,
                    0x00f299734c5bded9,
                ])),
                Felt::new(BigInteger384([
                    0x49bd840ebd979d55,
                    0x65b37f6778102cf9,
                    0x39bd92d7a68e7754,
                    0xecd700fb8d88f69f,
                    0x2cc517b6ba68ad43,
                    0x016cd130f3f69722,
                ])),
                Felt::new(BigInteger384([
                    0x6b0093c63632688f,
                    0xb7e5302dd4d3812d,
                    0xde3a8e96078b8db8,
                    0x19010388393f5b50,
                    0xb119f12ea019e0d6,
                    0x0089de10708dfd27,
                ])),
                Felt::new(BigInteger384([
                    0xce1908528ba8bffe,
                    0x806b2bef8b8d063c,
                    0x72296158a9362525,
                    0x137c6f45e1d0a997,
                    0x42b8bc9c95e9b5f3,
                    0x01962a548d005ae6,
                ])),
                Felt::new(BigInteger384([
                    0x12f42383d808b121,
                    0x59f35d91323eb91c,
                    0xf89abfc6a0539f2b,
                    0x18ddc4f31f583921,
                    0x0a4aa20786f4d231,
                    0x008bc2dee609df0a,
                ])),
                Felt::new(BigInteger384([
                    0x993719c3767f9429,
                    0x710b69bb7f52840d,
                    0x897f55bbc9b3b938,
                    0x50db1783d7f93eb0,
                    0x0e28ad42c34344b7,
                    0x01a0751608717a42,
                ])),
                Felt::new(BigInteger384([
                    0x3cfef250fb8fffe8,
                    0x23658911b42f5efd,
                    0xa35a09d5c3ee7b0a,
                    0x7fdc2739595851c1,
                    0x35d03350d46ada4b,
                    0x016e027918c098d8,
                ])),
                Felt::new(BigInteger384([
                    0x5ac040769cf3de7f,
                    0x662a3d0246c26745,
                    0xecef130f66bcf885,
                    0x984905f0043a62bb,
                    0x0e72276ddab4896f,
                    0x00b36cc834eabb69,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x7c80184c31e098c0,
                    0xeb60343181b74fc9,
                    0x2bb58790821aa9c4,
                    0x264c176c502d687d,
                    0xb463233555b148c8,
                    0x01015589b6f2451a,
                ])),
                Felt::new(BigInteger384([
                    0x881a24b8659b1c42,
                    0x3b7464bff5ea2bde,
                    0x87ead18f887724c3,
                    0x0c0f1843aa04d8e3,
                    0xb4a4e0583412fb9a,
                    0x013448558f68f60b,
                ])),
                Felt::new(BigInteger384([
                    0x3f4932cbcc4cc5a1,
                    0x27db183e0888a09c,
                    0xca37f8c868e8b530,
                    0x1d83db0c1a3eb65f,
                    0x797d23d553b304ad,
                    0x00d223f2294f1d1b,
                ])),
                Felt::new(BigInteger384([
                    0x6e43fa6d7f4fc394,
                    0x0f9a1437fda86cdf,
                    0x975c64ede359d091,
                    0x4d3478544613a3e1,
                    0x753a7dc66144d589,
                    0x0048694010f28a79,
                ])),
                Felt::new(BigInteger384([
                    0x4890032fd12a1f29,
                    0x552662b5d6be3657,
                    0x536cd25019b3fb2e,
                    0xea5c95efcd16c1f3,
                    0x70fe3751d8a4b627,
                    0x00d821399414ee14,
                ])),
                Felt::new(BigInteger384([
                    0x7029c52eb732b635,
                    0xf0d0226f1180578a,
                    0x99616ac58235f405,
                    0x23b9d1d792bd338a,
                    0x87484e16ec17dcfd,
                    0x00461b65845ad5fb,
                ])),
                Felt::new(BigInteger384([
                    0xa654ce0c9dd49619,
                    0x5e23cbcfd3281ffa,
                    0x1d471679f5597674,
                    0x708f7dce9b8c1d2e,
                    0x89a5debb9653f712,
                    0x00382c6992bab1e4,
                ])),
                Felt::new(BigInteger384([
                    0x9822860dd7991ffc,
                    0xfd8d0df062d7f94d,
                    0x7b30d0f8c5990891,
                    0x75a1d0e858ecbae5,
                    0xdfa6d68a0b05cb2a,
                    0x00247c9e745eec60,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x0515788a3db805db,
                    0x9f408e0a16f6920d,
                    0xcb9e67de90cca5a9,
                    0x571759494f20e7fb,
                    0x0bad2da9e4c0e73c,
                    0x00735a84c23c065a,
                ])),
                Felt::new(BigInteger384([
                    0x8f4e47f1a48d49ef,
                    0x8b6eff15033baf26,
                    0x1e93aa85d761b373,
                    0x5ce72cb42b37ce4c,
                    0xa1cfb4805fd19464,
                    0x01479f982136e1a1,
                ])),
                Felt::new(BigInteger384([
                    0xc314019e224f558e,
                    0xf541aea741ec47f3,
                    0x7489f25f7476a844,
                    0x551df46de3083f7e,
                    0xc755b752d3285cce,
                    0x0009e921bf3d0e75,
                ])),
                Felt::new(BigInteger384([
                    0x08fbd1c5c0737187,
                    0x0d6406f0b0260777,
                    0x12a3db389af957d2,
                    0x0548d26e5c4dc5d3,
                    0x082c2865eb85fb5c,
                    0x012886ff55a85480,
                ])),
                Felt::new(BigInteger384([
                    0xfe31ca874abc0a0c,
                    0xb65ad3f69ebeef6c,
                    0xdac7da3eaa1ec59f,
                    0x57e0ba23fe7399ae,
                    0x6e5211b92d4d4698,
                    0x00c28e4c08ebcdf0,
                ])),
                Felt::new(BigInteger384([
                    0x08ca0970f5cbbd1d,
                    0xf53da7c9daf82962,
                    0x8a143e45f2ee98a3,
                    0x7052e8d14ccd6c7b,
                    0xa6836641de7c0977,
                    0x0049b531b689bf72,
                ])),
                Felt::new(BigInteger384([
                    0x54a13f5a0f102e2a,
                    0x1c31811be551764f,
                    0x05e3ace45661b747,
                    0x0b251327f58e8b8d,
                    0xa9e307e423b8d52b,
                    0x003589aeccfbcae7,
                ])),
                Felt::new(BigInteger384([
                    0x0f11d70925a36ad3,
                    0x714866ea760896b9,
                    0x98fe3e9d063ba773,
                    0xccbc30e214cb3ee2,
                    0xc567a0ad4941b8e0,
                    0x0047db48a967df11,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xb2e117f2514c16a4,
                    0xc92023f0a54ec488,
                    0xec0f65e65461ce99,
                    0x243e58c14fd1eed2,
                    0x2273a1619c922bbf,
                    0x0092f665100c38ec,
                ])),
                Felt::new(BigInteger384([
                    0x47a972e65a9a439a,
                    0xc3fd2bb8d80c6ed1,
                    0x263192162dd8040d,
                    0x323d7b3685069305,
                    0x88ac130bbbd36370,
                    0x014ca6baca541848,
                ])),
                Felt::new(BigInteger384([
                    0x1d6709bc2e53a3e1,
                    0xe975b89a4fa38f0a,
                    0x82371c74ffd1c3af,
                    0x79d353fd8c794dbf,
                    0x3b4ceac4ffff2fb0,
                    0x00a66b510b262d26,
                ])),
                Felt::new(BigInteger384([
                    0x69430aab27dbac33,
                    0x13b603a43a4236dc,
                    0xbde992c8102e9550,
                    0x8c2f95f182d9fbd9,
                    0x054bf675b0da52b7,
                    0x0073c854cf331b00,
                ])),
                Felt::new(BigInteger384([
                    0xb167be776e6c8ce4,
                    0xf50ede6c86545540,
                    0x8dba6f5f840f337a,
                    0x9e31774cdee3c7d7,
                    0x7ffa4e075cd7e258,
                    0x011856080cba44ff,
                ])),
                Felt::new(BigInteger384([
                    0xeba75344d41d86ff,
                    0xd629414b1164e119,
                    0x64726a3f7f720bdf,
                    0x3278b63abf17d397,
                    0x9d18ccdb4d5f3381,
                    0x005cc88e02978498,
                ])),
                Felt::new(BigInteger384([
                    0x76f494a933e117a6,
                    0xa5b5dc2d2c69dc13,
                    0x9c2035a48bee6f26,
                    0x0db0d9e141087cdd,
                    0xd13d465a9859bbd1,
                    0x004fbdce39131e5d,
                ])),
                Felt::new(BigInteger384([
                    0xb04d7b195659ae7a,
                    0x508333b350090897,
                    0xa8b34c0146e6046b,
                    0x8a7eebefde2d1ee1,
                    0x269f104a7e6ab3a0,
                    0x01677364f264d25d,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0xbf3683b73ff486a9,
                    0x611df28290731ba0,
                    0x4d262acf17b604be,
                    0x83c2f1d3ec282e22,
                    0xd9a5fd9dbb2e962f,
                    0x00e08a057196af81,
                ])),
                Felt::new(BigInteger384([
                    0xce1ec37fd1eca16f,
                    0x819ce5d053ee7f01,
                    0xfef6507a5d624e0b,
                    0x9f1051f9e6825fef,
                    0x56b37a0883e08a5f,
                    0x0133c7b3eef973ef,
                ])),
                Felt::new(BigInteger384([
                    0x8f4f75f77361da28,
                    0x3ecf4eb3d58d44bd,
                    0x07437ed7c6485db6,
                    0xc290798ebaa48eec,
                    0xb113ea1b1b0c50f8,
                    0x003241b2648e04ae,
                ])),
                Felt::new(BigInteger384([
                    0x7533ebece6e02cf3,
                    0x633e37f2156e297c,
                    0x50bb09774d42935b,
                    0x849089facc315f6b,
                    0x0f75976bd30e47b1,
                    0x0074030c6d0db8d7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x62f296a47af5c3a1,
                    0x5db47e8a9c8aa120,
                    0x4fe27f3bded2ccf6,
                    0xa0451f714d6c40eb,
                    0x9288ceddb55156b5,
                    0x0156250d0e7abe37,
                ])),
                Felt::new(BigInteger384([
                    0x8176b399f4e26d62,
                    0x688660f8f13c0337,
                    0xddc72fc8f6400372,
                    0xbc0b46cf49d23e64,
                    0xba4d90470171cf52,
                    0x014a19231bff3f7c,
                ])),
                Felt::new(BigInteger384([
                    0xe100590af8215150,
                    0x8ae6410c981f1292,
                    0xf83c35f366722b72,
                    0x608ace4c203de8b5,
                    0xc237c94f144b58e4,
                    0x014ef4a22b90660e,
                ])),
                Felt::new(BigInteger384([
                    0x7511370f92df1e2f,
                    0xd20a476da35840d0,
                    0x81cbfb35008d6f34,
                    0x10ca5f4947b5f097,
                    0x73944ffa6045ed17,
                    0x00fef1fe86c703f0,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x5db2be20344f5f39,
                    0x6cabf6a73cbe7951,
                    0xeb490d0f45b015db,
                    0x8fc1b2cb63b1bcd1,
                    0xdfb37342dfe69e8e,
                    0x017fc37ab713be92,
                ])),
                Felt::new(BigInteger384([
                    0xa2ed3f27afe01214,
                    0xb5f9693992ca10ed,
                    0x956bf1e5974ce8d4,
                    0x8e679e42db7b15fe,
                    0x83e21951fe06015e,
                    0x00fb658abb207152,
                ])),
                Felt::new(BigInteger384([
                    0x335b70592f91302d,
                    0xc127990e0001bbe4,
                    0x1a9ee50261061735,
                    0xee5f49b4a26e4426,
                    0xccf2bf024ea35432,
                    0x00342d13257afdcd,
                ])),
                Felt::new(BigInteger384([
                    0x1594558e4440b5d3,
                    0xd60f553243230ea9,
                    0xa24e300dbd2fb30f,
                    0x33b0d38c39376df5,
                    0x7d52b5a3e3d4c272,
                    0x002cddbe616b1e4b,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb2dbf2cf3c30e93b,
                    0x175103ca0aa24725,
                    0x2a543466cb8cb3c7,
                    0x85b2f7f8b0f7144f,
                    0x958adb93df4d944e,
                    0x000802b3d8478f10,
                ])),
                Felt::new(BigInteger384([
                    0x6bd0c86cb066f66d,
                    0xb9f73027dda5c82d,
                    0x503c01af5a2d9cc3,
                    0x0157f44d64a72cee,
                    0xb99007411ac6174b,
                    0x0142ea5e0caa87fd,
                ])),
                Felt::new(BigInteger384([
                    0x89c031833c455cb2,
                    0x2b913da7b615f4bc,
                    0x7581b9f4f2916f89,
                    0x3fab0fcd7e4ba322,
                    0xf3d5a4b5e4cb8cb3,
                    0x00785ad38e90211a,
                ])),
                Felt::new(BigInteger384([
                    0x591c07b44c1c58c5,
                    0x6b0d493d06bfde19,
                    0xf440e3c182ddacd4,
                    0x0e3f251681895b0e,
                    0x97ae77d02de19515,
                    0x01301c3729ef6762,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xfbc30630f32a75c6,
                    0xede669e27f5a1da6,
                    0xec0c0a2845fc8b8e,
                    0xd44e63610f1990df,
                    0x95db2669680456bf,
                    0x00e8b519b17a7509,
                ])),
                Felt::new(BigInteger384([
                    0xaa2f4e3d2965e019,
                    0xbdd377c08b890f8f,
                    0xd3fb39b359eeda9c,
                    0x305da1490b82282f,
                    0xab952f2c036c98a1,
                    0x0187d74f5669f62f,
                ])),
                Felt::new(BigInteger384([
                    0x4a7ab88077be7269,
                    0xaa3d6c692e406460,
                    0xb97d24e4aa4b4378,
                    0x7289062c138b09be,
                    0x5cadaa88d6aff8db,
                    0x01617e0753d879e7,
                ])),
                Felt::new(BigInteger384([
                    0x7b338c845194f4c5,
                    0x2466ae9be3c5aee6,
                    0x7e92efff4258f812,
                    0x9c1e9d9c97dad2e1,
                    0x5afa23434b8439aa,
                    0x014f24928e37fc48,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x25b5f6be39b4cfce,
                    0x7aa5d1a50f98a947,
                    0xbc21e2b085dc4e84,
                    0xfafbcc9715330f67,
                    0x2ff3a2f42a8f2fac,
                    0x013c7db9a9d49863,
                ])),
                Felt::new(BigInteger384([
                    0x1ebc9a08b00f4d47,
                    0xcd2987eb78b79bdf,
                    0x7d37ed014c00ebab,
                    0x1f49ff771e10decf,
                    0xee738080fa2ccca0,
                    0x0064224461556c3f,
                ])),
                Felt::new(BigInteger384([
                    0x8862e37b73219435,
                    0x4d3064f63e994956,
                    0x5104e3e3e31c0952,
                    0x78d78bbc06421c02,
                    0x6d97f7525bb40260,
                    0x01a019a6b61c0b86,
                ])),
                Felt::new(BigInteger384([
                    0x58df5833f66c2594,
                    0xf46109def794d5d4,
                    0xc9a3525a986cf4b7,
                    0xbdd3465de5c03d54,
                    0x694d6415740c1d4e,
                    0x008c727d19bd9983,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x85b3fabc289067a9,
                    0x2cd54a6033b6ee16,
                    0x51172228a501c9c9,
                    0xcc27d6323fead1cb,
                    0xfdb18b9b76ee3f75,
                    0x017d5fa59a4add33,
                ])),
                Felt::new(BigInteger384([
                    0x1edb21ccd13abd88,
                    0x0527ea7e01c12574,
                    0x16f7c2e8a9d2f067,
                    0x1410503b15291112,
                    0xdab39ef8aad437ed,
                    0x008f5b1d0659edfd,
                ])),
                Felt::new(BigInteger384([
                    0x773d7c79e431cc6b,
                    0xc7ade7babaf612e5,
                    0x7d1dabb5b4c46cff,
                    0x64048fd56cdf4f27,
                    0xb0f64ebfc2478351,
                    0x0060f36e592f3764,
                ])),
                Felt::new(BigInteger384([
                    0x7327322590dd0976,
                    0x5fe7578fa61b5dc3,
                    0xa5ccf3c014c04231,
                    0x8086dc6ab1eb5a60,
                    0x4fb6ba91adbd2e98,
                    0x014467433696f714,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x10450b64ff6f42d1,
                    0x690981da234d44d0,
                    0xb5be0a1fdad314bb,
                    0xff276e7103347d75,
                    0x388bdb6153e6fec4,
                    0x017c1079ef0406a0,
                ])),
                Felt::new(BigInteger384([
                    0xdb0a9dfbd1ea54e8,
                    0xa5b1ed708f30f445,
                    0xdcd23ee1a581df53,
                    0x61cbc80922a5aa3e,
                    0xedf74ffa61160fc0,
                    0x0003e0297ba9bdb5,
                ])),
                Felt::new(BigInteger384([
                    0x3889406548d93eed,
                    0xede843b9db87cd2c,
                    0xf77d81e883d415a6,
                    0xf5591cffe41d06e6,
                    0x157357f28efd3daf,
                    0x013e5de0b81dbc6f,
                ])),
                Felt::new(BigInteger384([
                    0x557d04684e52d44a,
                    0xa4f696d63a8860d6,
                    0xfdcb212b908ac502,
                    0x44dbd5d5a50ef458,
                    0xecfa9721d8bd69c3,
                    0x016412d947bc4f80,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x6e49943c89dfb870,
                    0x42f928aa838d702b,
                    0xcc801b7944b4ccde,
                    0x4017559014212c50,
                    0x4261d094071cdf39,
                    0x014d4e9ca533f4d3,
                ])),
                Felt::new(BigInteger384([
                    0xae88dab73defb180,
                    0xef1e2856c419dccf,
                    0x3af1ad3d5e35c1c2,
                    0x0cc6d5a5101073ec,
                    0xcd562a6891e3332f,
                    0x00238d410c1fa049,
                ])),
                Felt::new(BigInteger384([
                    0x332a3fb22829bd94,
                    0xfe983df5bc4ec331,
                    0x09dbb6904eaa2b12,
                    0xadd02f87e156cbc9,
                    0x5b2455cc08ca091e,
                    0x0050ac55e3a1d40d,
                ])),
                Felt::new(BigInteger384([
                    0x83599445e0123482,
                    0x229188273269b567,
                    0x75bfc15b130802c7,
                    0xb85bcb9a9df3a15d,
                    0x4e3477c88bc263f9,
                    0x003e0273ced9b612,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x9149a129024fa41f,
                    0xb8be4023c38b4a90,
                    0x1b3e6734095564ef,
                    0xeff42a4f036f6ee2,
                    0x16d9b4cf3910ddca,
                    0x00ca1327337a7218,
                ])),
                Felt::new(BigInteger384([
                    0x79c1cc22f7aeebfd,
                    0xf71ffaaacf85bc89,
                    0x4c2baf978c8eb62c,
                    0x02b9e610648bf30f,
                    0xe5f423fef9ab86c0,
                    0x009a47c2bf531c7e,
                ])),
                Felt::new(BigInteger384([
                    0xfc3ac0f699f8b69a,
                    0xfe762eaa43156fab,
                    0x494c8a956ed40fd6,
                    0xb6c20875ae20c0fa,
                    0x0a6e02a0f3e739ce,
                    0x011a7fd3ab482ee4,
                ])),
                Felt::new(BigInteger384([
                    0xa505b3765aedc0da,
                    0x6d756c9215a7c350,
                    0x1b90cd3d2018b3fb,
                    0x3cca90f836df0f47,
                    0xe6acff97c74fba71,
                    0x01446460ebe23ffd,
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
            vec![Felt::one(); 8],
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
                Felt::new(BigInteger384([
                    0xa7d872d777c1c234,
                    0x7e05cc8d5fbd37dd,
                    0x28fbc4a88ca2bd94,
                    0x4d1dc5bd075ed6d0,
                    0x6da67b5988e7df14,
                    0x00c6f95139e19378,
                ])),
                Felt::new(BigInteger384([
                    0x52aac8b42b6a5ec7,
                    0x17dbd78546561419,
                    0x9e3ed6cd70149034,
                    0xedfcf6d1ca8d4d14,
                    0xd67fdc74dae9c0be,
                    0x00b849340ebaa5ab,
                ])),
                Felt::new(BigInteger384([
                    0x91eb065d0b9b10e4,
                    0x7ad86dfc6b37159d,
                    0x30b80832b85e35a6,
                    0x06966d1f24af0b99,
                    0xa930b52f6616af55,
                    0x007554b22c1df5cb,
                ])),
                Felt::new(BigInteger384([
                    0xbe74cd001647a568,
                    0x64d64bc705fde6b5,
                    0x2234a7abe0faa16d,
                    0xc0676a890151a6be,
                    0x78474ecceebd7741,
                    0x005a01b6b44556d3,
                ])),
                Felt::new(BigInteger384([
                    0x01df9d6bb7c7b0f0,
                    0x7227bd8a94eb6cc2,
                    0x13a7b00cf76fe541,
                    0xa1c55dafb6741413,
                    0x009f5afd66b63600,
                    0x016b4826e4a9ed0c,
                ])),
                Felt::new(BigInteger384([
                    0xc17e612405414a2f,
                    0x53605634e9123e7f,
                    0xbdf228f2de0305dc,
                    0x97823ddccc9f304b,
                    0xeb1f2fb8da378acf,
                    0x006c15221bca4c9e,
                ])),
                Felt::new(BigInteger384([
                    0x648ae3d06b6a8db2,
                    0x0acaee4e024bbc4e,
                    0x2e614262d2fdb255,
                    0xf313d2e3f018de0e,
                    0x6cee50675fd9d327,
                    0x006cf6924ade9f7e,
                ])),
                Felt::new(BigInteger384([
                    0x6b51c3d82c5acf08,
                    0x2edc45247719284c,
                    0x75bedce0ca68852b,
                    0xa797f53edd26fb46,
                    0x8dee85d94a850db6,
                    0x011afc757ce1da68,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x808e043e86cb39b1,
                    0x0efc89b283787fb4,
                    0x2f6c0502def418b1,
                    0x65778cb9b4ff5bb5,
                    0xfcf7c44ef69bd282,
                    0x00b1a3a26766555c,
                ])),
                Felt::new(BigInteger384([
                    0xe27c42037883b187,
                    0xe52adb69a87cffa5,
                    0x6fdff6cef7c5fed5,
                    0x9c570fd81e4caa78,
                    0xa697840c8823c96f,
                    0x0142af8f0885ef90,
                ])),
                Felt::new(BigInteger384([
                    0xdc24df5572df41a1,
                    0x53297b36e75322f3,
                    0xccef4cb565a723ae,
                    0xb3a7b8f709a9d682,
                    0x3404a379ab5a94be,
                    0x01985f763869f749,
                ])),
                Felt::new(BigInteger384([
                    0x6d2be7841296c8ae,
                    0x8c2ac9b0859add6e,
                    0x4833d37cdaf76cd3,
                    0xc935577c24cf5946,
                    0xb196278ba3db68fb,
                    0x00b6d5154ee25449,
                ])),
                Felt::new(BigInteger384([
                    0xd2483c51a59a17bf,
                    0xc45db006e61e8ebb,
                    0x94662cba3b4abac6,
                    0x78519f115f59847c,
                    0xcdd146b7339f8b71,
                    0x008cdd99484a8202,
                ])),
                Felt::new(BigInteger384([
                    0xc49324695ded0cab,
                    0x332d167d1373f9e5,
                    0xab5bcd735d49111c,
                    0x0487c5dee0623fb8,
                    0x871c58eef1527b0b,
                    0x0134569dfa30780a,
                ])),
                Felt::new(BigInteger384([
                    0x5af6b4428f8aafd4,
                    0xc797ce44b3dc49e5,
                    0xa51d516886d1b270,
                    0x08bc04621e0dc3c7,
                    0x8d71d091e17ec590,
                    0x015f203803c47541,
                ])),
                Felt::new(BigInteger384([
                    0xd03375590e3bd10e,
                    0x2538b03a03cbc0dc,
                    0x69aa2bded4cac8ee,
                    0x3cbfeaa14b61924c,
                    0xa31b5af2512d6fef,
                    0x00d9acb044e4d2ea,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5dd9ea1e4b8f8a17,
                    0x30d79f52df659ef2,
                    0xd6b95cf4dc200a39,
                    0xe3b88e68a79478f0,
                    0xc6e7eee5c22d9d73,
                    0x005e95c8201bd17b,
                ])),
                Felt::new(BigInteger384([
                    0xe49c80538966b247,
                    0xe182ae8b39752877,
                    0xa6900c84382e303e,
                    0x4e441b92aa959a5b,
                    0x18f55248ffe6bc4b,
                    0x00ea674a50e27efb,
                ])),
                Felt::new(BigInteger384([
                    0x57e7a83e5ea36a96,
                    0x22df8063c7205a0b,
                    0x9f9500ea3953fb17,
                    0x1733671e59a25f60,
                    0xc2a43ec8918c0bc2,
                    0x00ea9370f3af06f1,
                ])),
                Felt::new(BigInteger384([
                    0x65f71b4c33752b7a,
                    0xb0e2249eb60cfb42,
                    0xe6ce1730da246132,
                    0x06f9aab3ca39dad2,
                    0xacedf716740c92eb,
                    0x00ca3b4293d6e525,
                ])),
                Felt::new(BigInteger384([
                    0x6a8d8233559aebdf,
                    0xad7872386c053e1d,
                    0x6f60afb4df7a8a6d,
                    0xb3891e4a9aced488,
                    0x6dc8a922fb6b2126,
                    0x00b74d6855b834e9,
                ])),
                Felt::new(BigInteger384([
                    0x0aa850f2dc3db522,
                    0x920be4b2a7716547,
                    0x22f3b02c0acd410c,
                    0x1343c994450913ce,
                    0x3e52e924e7f9007c,
                    0x018a687746edea33,
                ])),
                Felt::new(BigInteger384([
                    0x4eb9a5bfb4458ded,
                    0xa31822036787ce72,
                    0x0f790ab3e8092aa1,
                    0x5f3e4791f2af657f,
                    0xf222d18f64381e41,
                    0x00499de69b41bf2a,
                ])),
                Felt::new(BigInteger384([
                    0x6a0aab4baaad142d,
                    0x9daf897143cfbf71,
                    0x3bc1fff4f010024e,
                    0xf1e617f6b8c84c88,
                    0x391ebcdeb378ef21,
                    0x010c7d55d0a16561,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa09e94a387eba77a,
                    0x625b171d02a8f5e5,
                    0x28e56d686b146041,
                    0x10aa358dbaa3342e,
                    0x1c88d77c765227da,
                    0x015e211499d61ae3,
                ])),
                Felt::new(BigInteger384([
                    0x16b5ff4343f78883,
                    0xe16497eddd9f35db,
                    0xdcd485f5ef6ebef2,
                    0x54f7dc1ac6361b5f,
                    0xfd2a3ad367d001d8,
                    0x011ed91988195185,
                ])),
                Felt::new(BigInteger384([
                    0x87e899e418a3fb34,
                    0x73cb329d8819d018,
                    0x98da64780644d44d,
                    0x6ed66bb528c3580e,
                    0x1f939c53057a37b6,
                    0x00a9b4704040f516,
                ])),
                Felt::new(BigInteger384([
                    0xca47d5b2d611b26e,
                    0x1bcbe116a79bd1e4,
                    0x4bc12ac41662361a,
                    0xa54f373c8b8745c7,
                    0x511b738bc8275336,
                    0x0048c3176b07abd0,
                ])),
                Felt::new(BigInteger384([
                    0xcf187dc9cbf8007f,
                    0x732cab04b4662e03,
                    0xd3579d92629a2b84,
                    0x3bd085395c77efd4,
                    0x74ae24bfeddc45b8,
                    0x01777821e3e9817b,
                ])),
                Felt::new(BigInteger384([
                    0xb00e58bfc3e81c4d,
                    0xa21e3d353cf77843,
                    0xb5b82506f569b0fa,
                    0x864f7e73d477bd85,
                    0x84d10cb136885623,
                    0x00ebdb3e01e39301,
                ])),
                Felt::new(BigInteger384([
                    0xfb0cf88eddb9eb2a,
                    0x42f845d1ed79a2ab,
                    0xa720dcc8a360c7c3,
                    0x8d5b5313195a6283,
                    0x2ee45cc6b63db787,
                    0x0002f09e095babc5,
                ])),
                Felt::new(BigInteger384([
                    0xd252ce68b17bd50b,
                    0x6f13a87599c0e736,
                    0x2f27ba4d7f435b5a,
                    0xcd6bf744b765c529,
                    0x9476acc964c11022,
                    0x00b142a3c161290b,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x1b17b6550c01a137,
                    0x9829315a3467690b,
                    0x874bea407137f838,
                    0xfe2a38a8bee5bfc1,
                    0x96e0bfce6ced2297,
                    0x005bca87dfba3529,
                ])),
                Felt::new(BigInteger384([
                    0x55edfc0cf2406fec,
                    0xd75db8ce23c02876,
                    0x1e46b69b72167d62,
                    0xc0efba8cfa4a1237,
                    0x44886487c38db97a,
                    0x006c72049d407923,
                ])),
                Felt::new(BigInteger384([
                    0xaa797dcb82a719c6,
                    0x3837b39c54ed2dc9,
                    0x7c7f7e73402f962d,
                    0xc0447848c080f839,
                    0x3fba2ccb2fe9cc69,
                    0x012337c845f4e849,
                ])),
                Felt::new(BigInteger384([
                    0x58ecd70723822f49,
                    0xa36297b5fbd1d4bd,
                    0xac10f44194a4b337,
                    0x33af92be400e87a3,
                    0x00f6251ddc3f8781,
                    0x012335d9563ae81d,
                ])),
                Felt::new(BigInteger384([
                    0x1aa72ac2c6cc3240,
                    0x5d398dec288316dc,
                    0x28d43a24850b6da8,
                    0x2cf00d4985ed70ec,
                    0x083a981a3cdcbcb8,
                    0x017cb1059c8756a7,
                ])),
                Felt::new(BigInteger384([
                    0x7490b4ce76f509b9,
                    0xf5a1e868e8ec4df0,
                    0x37c17eef477ae619,
                    0xe7748ca6cc8fe840,
                    0x7bc58ce36e4fa67d,
                    0x0184fa6e2a70f7be,
                ])),
                Felt::new(BigInteger384([
                    0xbeddb4cf0c66a66e,
                    0xb1eece15645b00e3,
                    0xc5b19422a5947c83,
                    0x9097a7490dff701c,
                    0x1829b9ff613f9bb6,
                    0x00c309f30be9e5dc,
                ])),
                Felt::new(BigInteger384([
                    0x634a6b26e86f8d22,
                    0x057cb631bc10bd56,
                    0xcf1372115041d903,
                    0x1050f7da2a847bec,
                    0x8bb7d70a0c89441b,
                    0x0021a5cfb2486a74,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x40f5389547afa7fb,
                    0xd510594db0f566bc,
                    0x490694e0e8725636,
                    0xe4b5a59e01ad1296,
                    0x2f02e0f0b496da6b,
                    0x00e70aa741f1383b,
                ])),
                Felt::new(BigInteger384([
                    0x43e41000e9959ab8,
                    0xca6eb5838e1a05fb,
                    0xc5bc056a9cae359a,
                    0x7d53847ab2180aef,
                    0x46b1bec5280de2db,
                    0x000c863c2764edca,
                ])),
                Felt::new(BigInteger384([
                    0x8daaacf0943c90b9,
                    0x7e11b0fc8c1393d0,
                    0x819358165df64df0,
                    0x73cd91662b4c4a79,
                    0xe3caae9e678f3d5e,
                    0x014fdffb1d7d573f,
                ])),
                Felt::new(BigInteger384([
                    0x42e60365c4314df8,
                    0x1ef552ec4a353170,
                    0x6ce8299f3ffed807,
                    0x84bb33ea612f9304,
                    0x7fcad30e1f72aded,
                    0x01300b06f5784f2f,
                ])),
                Felt::new(BigInteger384([
                    0xc009e7405d1b6f0f,
                    0x1eb4cec25c63ecf9,
                    0x0915bb2e571f92fc,
                    0x36559ae2cdba515e,
                    0x78d45ddb8e5288d1,
                    0x0191bbfe646769a2,
                ])),
                Felt::new(BigInteger384([
                    0xfd8835bc2066752d,
                    0x97ea756ca0bd893c,
                    0xd52f4ae9b4012772,
                    0x5037ae9837eaddf4,
                    0x5d7a1f18d3901253,
                    0x0154eefd97e44cb4,
                ])),
                Felt::new(BigInteger384([
                    0x7d122b0b41f9f781,
                    0x2dc8642e213a5784,
                    0xbc42a58c3ab496c4,
                    0xc31a2efa1d0376c3,
                    0x0006da7f9d693c56,
                    0x0045cdbc8c0b5321,
                ])),
                Felt::new(BigInteger384([
                    0x7e2de1b125142ff2,
                    0xdbe4f7b574c721f1,
                    0x8079c2a84633f78c,
                    0xb31fc55817f01bf7,
                    0xc5d7b082bad423b8,
                    0x0095ae2e90dfc3e7,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x0ccfe91b6c232f32,
                0x6dbd01b49f5d08dc,
                0x8527a168ce99fbdb,
                0x4fd16d64588b68da,
                0x2aa7f36cc0886ffe,
                0x010c5c321a66d00c,
            ]))],
            [Felt::new(BigInteger384([
                0x30695a58fad8a080,
                0xf514ad75693df7ba,
                0x69cb1bcdc7ffdb0f,
                0x995fdfeffd48317f,
                0xf62c6ced5211d98d,
                0x0191b044ad4745dd,
            ]))],
            [Felt::new(BigInteger384([
                0xc487032f5801574c,
                0xa2d0f0dce2ad54cb,
                0x1eaeb1d5412980f5,
                0x2616945c19dd715d,
                0xe79ffb7aa3c36d57,
                0x012df990e1553b13,
            ]))],
            [Felt::new(BigInteger384([
                0x7c80347374f9951e,
                0x50db5d92751de228,
                0xc55f719ce12024e8,
                0xbad24737147e2bdf,
                0x1463f99aa01f8426,
                0x014529d685ac8ea1,
            ]))],
            [Felt::new(BigInteger384([
                0x2a32749aedbfa654,
                0xd00b4af024d4f131,
                0xc4b491a4e61ffd1c,
                0xa1813bef56378912,
                0x81dfd7883cca8aa8,
                0x015ee219fbc39c74,
            ]))],
            [Felt::new(BigInteger384([
                0x122e6cbd8ee34f11,
                0x598cc6727d030f1b,
                0xd9cc95617a23b845,
                0x72db21aff612e08a,
                0x8955ff55a412e4a3,
                0x0073f3c094a14b46,
            ]))],
            [Felt::new(BigInteger384([
                0x1dffb0132918dcb2,
                0x3c64bf0b33233f4a,
                0xb419137461a66724,
                0xa8e9ba4746e08e6b,
                0x64cc750bf5c91c4f,
                0x00789dc4cae6b37f,
            ]))],
            [Felt::new(BigInteger384([
                0xec38c208e9b69f73,
                0x10c8b508586a5116,
                0x84cc62afb9bf7e76,
                0xaffd8afec0a38d87,
                0xed931ed7d3662909,
                0x00b9b8b45a15b551,
            ]))],
            [Felt::new(BigInteger384([
                0xe815db935253e872,
                0xe8c0ce14a10b1360,
                0xf90c2615d57f9516,
                0x1110ed6396185dc6,
                0xd5aa892d15744a65,
                0x017d63c3744521b6,
            ]))],
            [Felt::new(BigInteger384([
                0x64293c9668ee3b26,
                0xa5e19e86c13a1059,
                0x31582bf009d4a7aa,
                0x9b0cb9ba1b19d8d4,
                0x123bbfbb7cf5b309,
                0x01140a31cc038758,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 8));
        }
    }
}
