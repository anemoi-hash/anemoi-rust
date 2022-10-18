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
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger384;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![Felt::new(BigInteger384([
                0xe82acf9cd22046c0,
                0xbaeeb6f4d8310f30,
                0xd7674d22b8c6f115,
                0x232cb0cedd1a85d5,
                0x95315d9a409e6b42,
                0x0164f4b0bd2c7069,
            ]))],
            vec![
                Felt::new(BigInteger384([
                    0x7cd95521692a8cb1,
                    0xfbe8768c469f2081,
                    0xcca0d58318049a75,
                    0xa72d5446ea747b61,
                    0x013197ada7d58cc0,
                    0x005a48eccd4950d9,
                ])),
                Felt::new(BigInteger384([
                    0x5fa5f2ff0b9467d5,
                    0x495bc18d35de9717,
                    0xe63dd3ba2e11dd6c,
                    0x30e68ccdc4345f7e,
                    0x4b252f8e784a9dbe,
                    0x00c568104141a05f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x88b153beaff74a92,
                    0xc0cc8f1ee69402d4,
                    0xdf80e8767f2af86c,
                    0xd877e872e69adece,
                    0xc83e2454b0a4ecb0,
                    0x0138cd42ad4888f7,
                ])),
                Felt::new(BigInteger384([
                    0xc3aa959f7b1d890d,
                    0xd87913f0225008e2,
                    0xb2e98f7123a1c687,
                    0xee9a0a4c5cf6b883,
                    0xa2f1b446fabeeb0f,
                    0x017e11b08ba842e9,
                ])),
                Felt::new(BigInteger384([
                    0x9bc620c3fd1236e9,
                    0xa916960697106041,
                    0x1b65dca0a84ca09f,
                    0xf2795d7ba84cf20d,
                    0x3572c4461f6e6717,
                    0x01ac8d5fa43c930f,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xe0b8fd45b5b45449,
                    0x9c219e17e8380b36,
                    0xfbe60f2a631eb80a,
                    0x25e514c54e6568f1,
                    0x47be608a51a37d8e,
                    0x000338477b4150fe,
                ])),
                Felt::new(BigInteger384([
                    0x8dc44a708ab94318,
                    0xf39016b59c310040,
                    0x882a1211e45145a1,
                    0xd8d730a5add60f1e,
                    0xb57022b6aa85f911,
                    0x0027288db7e64616,
                ])),
                Felt::new(BigInteger384([
                    0x941019a1ab93072c,
                    0xb2e7894b5dec4e5b,
                    0x6d1ee41fc7b391dd,
                    0x52bfda24d4cdd053,
                    0x7df9843765ff373a,
                    0x002e55a45d98cdbc,
                ])),
                Felt::new(BigInteger384([
                    0xdab950ad0fff99c8,
                    0xec7adb6a2c71fbfc,
                    0x999e1da14ecadce6,
                    0xe445e8a5cc90a91f,
                    0x0343cdb6d0644b90,
                    0x01a58b562a6d685c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x464d18b39bc026d5,
                    0x336c22606939231a,
                    0x5d822a226566a685,
                    0x2b757803ab7d2a0d,
                    0xbe136562024fee61,
                    0x0006e13ef51b41f6,
                ])),
                Felt::new(BigInteger384([
                    0x200abbeeaeaf3e08,
                    0x2cb93b31ec53bc54,
                    0xa6f8252c4f138d19,
                    0x3ff8603c869641b5,
                    0x4ee37c05bc11c1ca,
                    0x013822b45ce30bf4,
                ])),
                Felt::new(BigInteger384([
                    0x01dd2d65ff78ac56,
                    0x53d11f3c0b2c1368,
                    0xedb7d7888f837205,
                    0x578e684da21dcaad,
                    0xcca789bb833d023e,
                    0x001ecda848711c5d,
                ])),
                Felt::new(BigInteger384([
                    0x2160dede8aa8759f,
                    0x2eec9f4be8a5fac2,
                    0xf1d3abb703fb62d9,
                    0x2b93c33e99f07d66,
                    0xf775c2a1fd3ef6cd,
                    0x00bb775676b7d7ac,
                ])),
                Felt::new(BigInteger384([
                    0x8c932a0b382a071a,
                    0xa53aa7de1e8945f7,
                    0x3965d24a2ab1ac79,
                    0x2684669351522c4d,
                    0xdfb5d6fc963b1e27,
                    0x01988c2a5627af60,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x6b8f3291d6d45710,
                    0x444379bf7a3d21f8,
                    0x77b60161a5bb03b7,
                    0x590a6d08b356fd06,
                    0x00d3f34131531310,
                    0x007ae2a9959cacb1,
                ])),
                Felt::new(BigInteger384([
                    0xd041d35451df4870,
                    0x99c44c7bbe3501ae,
                    0x6b3ae86f8fa11363,
                    0xcd39344b752cda6c,
                    0x5637aedb859bd823,
                    0x018b3bce4ddbcfd9,
                ])),
                Felt::new(BigInteger384([
                    0x7b5bf0eddc9bd80b,
                    0xbcad0b852e620fd4,
                    0x4d85aef3d13b327b,
                    0x5bdda4a2588e6682,
                    0x0c882fab3f04dfbd,
                    0x008336a99e5a537f,
                ])),
                Felt::new(BigInteger384([
                    0x0ac7e81baabb9fca,
                    0x437181f0168971fd,
                    0xd392f7cd97465daa,
                    0x7a5ecf5532a580ce,
                    0x58db8533cf2270b1,
                    0x018a977d45044189,
                ])),
                Felt::new(BigInteger384([
                    0x01fe80ea10f61cc1,
                    0x05d686361aad2714,
                    0xddc46619531e7b06,
                    0xb2868f851c8e21ba,
                    0x7b5e08193f59d0b2,
                    0x003eab4947bc7cb1,
                ])),
                Felt::new(BigInteger384([
                    0xf57807e37067eb24,
                    0x0c0382a337a1d8e9,
                    0x29daa85fc9d880b1,
                    0xb3cad5b501a94a67,
                    0xd7dd47b13a4be4e1,
                    0x0170059984c5d024,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x9f2b3ce38e21f786,
                0x028467f028446642,
                0xb9cc11e235a25041,
                0x1f469265777ba470,
                0x6519b6308c8e91fb,
                0x00748bfb0648f46f,
            ]))],
            [Felt::new(BigInteger384([
                0x83b35967ae1f3764,
                0x1f7f8ae69af345f7,
                0xa2011fbdf88a368f,
                0x77c8e2b435d8e262,
                0x6ac75d111ce7e40d,
                0x000be639d3bcc53b,
            ]))],
            [Felt::new(BigInteger384([
                0x62d4850d9e06ac3d,
                0xb09527ee994e3d8f,
                0x4e8b790d1517a5ce,
                0x92f368ed1fc29674,
                0xa6083e6b397d4ff8,
                0x007ecb2e2fc16802,
            ]))],
            [Felt::new(BigInteger384([
                0xfeba230d9275edcb,
                0x87b63fc3e47b9554,
                0xa184c74c6b8748d5,
                0x9581645cb755c521,
                0x1cbf5c8f4d0b8c1d,
                0x0155bc43a97e34fb,
            ]))],
            [Felt::new(BigInteger384([
                0x459569cd9b041c35,
                0x79d8c85f6c91b55b,
                0x8ed89753fd0624b2,
                0xc061147c183fc006,
                0xc5692dce937204c7,
                0x0123a471e619b23d,
            ]))],
            [Felt::new(BigInteger384([
                0x67fbe35cb71d8ba9,
                0x5a2e0e07e6faac46,
                0x62ae15c9688c169f,
                0x0d254f09461b3382,
                0xe9e12bb1b19dfb5c,
                0x008ccaf4f704f6ef,
            ]))],
            [Felt::new(BigInteger384([
                0xf1d4f14dd68cd7ef,
                0xbb7c9f76d707a63e,
                0x7a0f74c0f9d82642,
                0xd272074d8dd63a95,
                0x10599656e91a2df4,
                0x0123b7930a567264,
            ]))],
            [Felt::new(BigInteger384([
                0xb0c69210ba465a77,
                0x4e034efbb64a02e1,
                0x7283a86440d9bcf7,
                0x502742d7945bf234,
                0x050c62217009140f,
                0x006f556a291b40f2,
            ]))],
            [Felt::new(BigInteger384([
                0x1fec7d1b8acb12b3,
                0x1888cf9e9456b654,
                0xab0478cd032f772c,
                0xbb4d14f6d21f04c8,
                0x53345baec86ed5f5,
                0x0119e194354ed580,
            ]))],
            [Felt::new(BigInteger384([
                0x8b446d833d887f91,
                0x8eef9fa26ec7788b,
                0x669cd3be546b4fae,
                0x3b0e42316a8504b9,
                0x2959bb0d56b6fd16,
                0x014f6a3d5c1afff4,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x28a87716646de305,
                    0x05f1e49b917f85e0,
                    0xa72e42480c587817,
                    0xad10a16e49f94768,
                    0x6e750a1b5a915abd,
                    0x0120586365ba36a7,
                ])),
                Felt::new(BigInteger384([
                    0x57d357f83ac106fe,
                    0x70583ad133241c44,
                    0x38b230ec8ff20879,
                    0x2b9820fb92507965,
                    0x57c46a181d2b820b,
                    0x00810c5ff2035128,
                ])),
                Felt::new(BigInteger384([
                    0xcecc55b71defca51,
                    0x35004fa692f7c2cb,
                    0x84d950a9a199619f,
                    0x9a7af59283600217,
                    0x1ea53d267af1efab,
                    0x006245293cacb935,
                ])),
                Felt::new(BigInteger384([
                    0x6cc0d68b35940ae6,
                    0x918d2778d2830659,
                    0xfe2c0a51871cbbc2,
                    0x1b6d1bf5291f7f61,
                    0x10e6a3154d0b66e9,
                    0x001a32bbc086d6bb,
                ])),
                Felt::new(BigInteger384([
                    0x37e06fa3b0d32c7e,
                    0xe958b94247013214,
                    0x25c8d91750c62d0f,
                    0xeea960c20cfda8f8,
                    0xeaff7f8b3b87f5b0,
                    0x0109c30e56e0b5c4,
                ])),
                Felt::new(BigInteger384([
                    0xc5544d01ea3c7c9d,
                    0xe9ec9dcc053d9532,
                    0x8b629cdf69e9ae0a,
                    0x49df5dcc58b8c1ca,
                    0x7d1dfe3f9bebc486,
                    0x0176de2545c4d9e0,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x04b0885c74877d79,
                    0x9b22f7d700105948,
                    0x4e164e9f7c9c4711,
                    0xb99881c7a9ca748b,
                    0x98cf8a00bd27f956,
                    0x01647cc3ef8d955d,
                ])),
                Felt::new(BigInteger384([
                    0xc5d95c9195cd6450,
                    0x890269bded07fb24,
                    0x5e127847c58ab3b4,
                    0x14674d3e558a1166,
                    0xe291b01ff95e4524,
                    0x0028807c4400dc14,
                ])),
                Felt::new(BigInteger384([
                    0x7d92252da3353bd5,
                    0x75182e87fbef8093,
                    0x1c9463133b555bc0,
                    0xe54c230b8c59e62d,
                    0x465e8b4bae2d8ed7,
                    0x01643f28fec7c10e,
                ])),
                Felt::new(BigInteger384([
                    0x8f725263a45b7d1e,
                    0xfba088865beaed3f,
                    0xf0df78f2aa8e99da,
                    0x558ed5291df040ef,
                    0xd4058f6d4a023e1f,
                    0x002cd3241b08aad2,
                ])),
                Felt::new(BigInteger384([
                    0xa9084338734efb91,
                    0x683d351b922c1bea,
                    0x295f609c149f1122,
                    0x9e4a86699a9e09aa,
                    0xd594e90226a7834b,
                    0x00988c08343518e2,
                ])),
                Felt::new(BigInteger384([
                    0x3ac5808e92df3564,
                    0x886c9d06706fa96d,
                    0xdf2330c052265216,
                    0x2324e2a247363dc3,
                    0x7bfd609876877fc7,
                    0x001f9b727df1549d,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xef7f4733aec97a4a,
                    0x19b65104379f3bb8,
                    0xd14f920dfbb68bf9,
                    0xe6249daea25897c1,
                    0x84c8678ea45a5771,
                    0x014a3e299e3ac150,
                ])),
                Felt::new(BigInteger384([
                    0x5d5853118c6f0951,
                    0x4ef6ce521c728cc3,
                    0x3dc0cb349f451d31,
                    0xe10a57f1ddf06c65,
                    0x7267807d17d9febf,
                    0x005a3e6a0713f1dc,
                ])),
                Felt::new(BigInteger384([
                    0x59e4af429a821620,
                    0x41a4de9bd9e2197c,
                    0x6e7b3149db19d54d,
                    0xa3637a51a444ede0,
                    0x137e10c50a917617,
                    0x00dc35a949812b1c,
                ])),
                Felt::new(BigInteger384([
                    0x824d07db5c420081,
                    0x75cf67d51de1dc00,
                    0x0b3bbb9c1bb5fd9e,
                    0x33f4643a626632f0,
                    0x386fb380a17c7da6,
                    0x00b6a05a05b1148d,
                ])),
                Felt::new(BigInteger384([
                    0xbaf1d86461d7f271,
                    0xcbbcc0a8e82c655b,
                    0xb657e9041ae87e86,
                    0x6c620ca149827d2b,
                    0x95652f22ec9608b7,
                    0x017c9b72fa51f6a0,
                ])),
                Felt::new(BigInteger384([
                    0xa5844aa3f704471e,
                    0xeb3338f98e88ba7b,
                    0xe51f9b42c931eb4c,
                    0x449a460be4102ba4,
                    0xa6cbe6dad62eacc5,
                    0x0027408c2cbd830c,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x453a63897f501a39,
                    0x2bd0974c9ff09eaa,
                    0x8f650a4c27c99eff,
                    0x4c4a3996a9bd3870,
                    0x1e91ff3cf60d6c0f,
                    0x01413ff6c5e55be8,
                ])),
                Felt::new(BigInteger384([
                    0x82a4570e6c117eb6,
                    0xd68b02dccc12ceda,
                    0x1e6f28d0a9561227,
                    0xad37b37202daa2e3,
                    0xe5ae125443668ab6,
                    0x00571b6f76ddf11b,
                ])),
                Felt::new(BigInteger384([
                    0xd3ca3f69ed5874ac,
                    0x4f011da4b446c9a7,
                    0x14582c0619ca1c9d,
                    0x45723a20c638af8b,
                    0xdd51a32773fa5353,
                    0x003f8807c8371b73,
                ])),
                Felt::new(BigInteger384([
                    0x9a8e7ba3bd89c217,
                    0xe9aae2cd23e1b97c,
                    0x90ccf4c8c2eaa124,
                    0x4532a85e116a6115,
                    0xf2386b40ce4fc6bc,
                    0x00d965eebeb1c1d5,
                ])),
                Felt::new(BigInteger384([
                    0xe304f66102bc06e3,
                    0x45c2a7727fde5e48,
                    0x83c05461b785d4b1,
                    0x3751360d3d292c1a,
                    0x41569a68737ff583,
                    0x013acea24765d604,
                ])),
                Felt::new(BigInteger384([
                    0xa19ecc63dd54b031,
                    0xf81711a831dad935,
                    0x73d85424ec0d8c41,
                    0x6de6f331a75b5658,
                    0x0e8b8b42c74de022,
                    0x000f74a18f798c48,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x88392d30143dbef1,
                    0x618657352189efce,
                    0xb6fe7e7a05014149,
                    0xed882367a19a86c6,
                    0x2c08df3cec414b25,
                    0x0161f326fe716e9c,
                ])),
                Felt::new(BigInteger384([
                    0xd7ec30d187759985,
                    0x9df68afea8f260dd,
                    0xf65df99980712c3c,
                    0x7024a3dc52687a0b,
                    0x139a08f1f70cec4e,
                    0x00639eb6cf97bc4c,
                ])),
                Felt::new(BigInteger384([
                    0x73bcaa1581a0e87d,
                    0x3c9c3b384195fb89,
                    0x751679c643d88d9e,
                    0x7974a52c0a819398,
                    0x78eabe5b58071a37,
                    0x005d254c4fda7014,
                ])),
                Felt::new(BigInteger384([
                    0x6c2fafbd1d120d6b,
                    0x8d2f14d5d815c2cb,
                    0x843e1298c7ec6f8e,
                    0xaa1ab0a12b8771b6,
                    0x3ec6723beb11d1ab,
                    0x008dae03328d3c9b,
                ])),
                Felt::new(BigInteger384([
                    0x3b6d6947651c86d5,
                    0x6e042a98b15ca007,
                    0xf8d2b0c4a8be796b,
                    0x5b01db878464cc9d,
                    0x41e7aec2e063384f,
                    0x011323f485ddf5fa,
                ])),
                Felt::new(BigInteger384([
                    0xe744bcd6aedb1cef,
                    0x1afbebc57d364fab,
                    0x6d3621106980852e,
                    0x75ac910586831269,
                    0xde435cc3b01aea4a,
                    0x000f8216a67d8b30,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x5b86ad949eef8b8c,
                    0x3ea85e686e851fa9,
                    0x5f525812dd4c5031,
                    0xcd35d44a81d227cc,
                    0x2ba5a0361319498f,
                    0x00716d3fd27b73a0,
                ])),
                Felt::new(BigInteger384([
                    0x178c6ef6e9bd89bd,
                    0x587677c346672bca,
                    0x21e46506d2e22495,
                    0xc4e6225fd4f875d6,
                    0x5bc245a21e9a87c1,
                    0x00b1a6129d2526f5,
                ])),
                Felt::new(BigInteger384([
                    0xeb8d875534cf75cf,
                    0xff202db50b86b66b,
                    0x3be5f095f12db0a3,
                    0x7a3269c326248f5d,
                    0x618526e3e49abf33,
                    0x004fc89d33e8cbf2,
                ])),
                Felt::new(BigInteger384([
                    0x7e81c36a2f6edd05,
                    0x2bca59720abffbd0,
                    0xa856c442432857b7,
                    0x23fb3452988f98e1,
                    0x308ac8ff41e50a6f,
                    0x0044c865e52fc318,
                ])),
                Felt::new(BigInteger384([
                    0xa09a32448a3e9e2a,
                    0x841fcc7dca5c128a,
                    0xadd30c7edb223f88,
                    0xbfb7a732b88e5b51,
                    0x143bf5ff3c95d91f,
                    0x001eddc5c4007699,
                ])),
                Felt::new(BigInteger384([
                    0xd69a327d3749262d,
                    0x4526fa00d8eedc2e,
                    0xcde59a8bc82822f8,
                    0xdbdd5154fcd20af1,
                    0xc439966a939d2f2a,
                    0x018cba93093100a7,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger384([
                    0x9a785cde5e67fe7a,
                    0xfb31d2525c002585,
                    0x8e74dc82b95a6e8b,
                    0x2732d931dccff8b1,
                    0xfe77dc08c08a92b0,
                    0x014f61dfb0b14460,
                ])),
                Felt::new(BigInteger384([
                    0x009fd8a479b9c348,
                    0xd1f3f1ebeebea9d4,
                    0x8b7cf49c7aadba1e,
                    0xcf9f40fc998ba085,
                    0x26b978f06c8141e4,
                    0x013e96485c7851bf,
                ])),
                Felt::new(BigInteger384([
                    0x1fbe53ea8a0f1a0f,
                    0x1203df653a1355ed,
                    0x5003565823e8c168,
                    0xa1f2750f872dfcf6,
                    0x3d179169e0f72a9c,
                    0x002b19d3819230a7,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd6895beec6708e1b,
                    0x65ba12b5f93abe9c,
                    0xfa9c067cca9c6a24,
                    0xc744e2e30a4a9811,
                    0x6316b19d31952780,
                    0x00c955132c82dc61,
                ])),
                Felt::new(BigInteger384([
                    0x0a23b700add429ec,
                    0x27fbd3cf4b99f995,
                    0x0499a6d5bbcc005d,
                    0x2da2fb8e05628d6e,
                    0xd8b8538a3fb67c2d,
                    0x00f584a3a2a3c230,
                ])),
                Felt::new(BigInteger384([
                    0xbcfdd249f6ad06f5,
                    0x8c1cf9e29e600534,
                    0x9a24856164187f55,
                    0xa16c1d481486b715,
                    0x4e0f9bca42a682ce,
                    0x011cce2a0e8c8238,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x8e659ce1b091a054,
                    0x00a9d85c8250f0c4,
                    0xef162eda42715d75,
                    0xc43d20c1930667af,
                    0x771e4a607bf51d1d,
                    0x005b42cba9d536bd,
                ])),
                Felt::new(BigInteger384([
                    0x1461649952ae6976,
                    0x4392c69b0793ea51,
                    0x461b13a4ef9d4919,
                    0xc0dd9c1cf4368359,
                    0xb4e8ec94318cac42,
                    0x013c664b97e39382,
                ])),
                Felt::new(BigInteger384([
                    0xe54e3649f5fcec98,
                    0x7a036223f548ef86,
                    0x7b67890d5ecab1e4,
                    0x5c1611065d2debb9,
                    0x2a96d6f4263a67a3,
                    0x0026b646edcd18ad,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x5fb69f7247e3f833,
                    0xdb3c365ebd621f89,
                    0x76f56a01230f5491,
                    0xf80c70170414c300,
                    0xb1a538238f981d7a,
                    0x01234e0a4b968704,
                ])),
                Felt::new(BigInteger384([
                    0x96bfbf18b42bcb94,
                    0xf24ee478b5275961,
                    0xd3c3d5f2b043850a,
                    0x43c0d6f96d25cb09,
                    0x8999b99e5fcd98df,
                    0x00ec8e69e99ea8a8,
                ])),
                Felt::new(BigInteger384([
                    0xee20d8262dc5cabc,
                    0x7b298972fdb10281,
                    0x48f35398442bd058,
                    0x75fcd9ce3ad1edc7,
                    0xd0828eedb7cac0fa,
                    0x00600a77ab79e459,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xd663a9fc682214f0,
                    0xcb8043b732c7bb60,
                    0xa28a6f827f2b182e,
                    0x10426ac8e0ae04a5,
                    0xb4b2c361b263e41a,
                    0x01830c8588858116,
                ])),
                Felt::new(BigInteger384([
                    0xb71f9579d5fd13c6,
                    0xd10820c9f38e909c,
                    0x7c8781a186b1050e,
                    0xcb3cc5a0955c6202,
                    0x18094defc72760fe,
                    0x001dfce22cbb90e5,
                ])),
                Felt::new(BigInteger384([
                    0x4931b5603d4acb59,
                    0xf9c9594fca91d681,
                    0xdaf78ea25f5422ff,
                    0x87ac36c9f9646391,
                    0x14cbc6d3d07b2972,
                    0x013e3c45f7b356fb,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xb3d69a4d34cd97a3,
                    0x8fca73c84e881dc8,
                    0xb4d26a06fe8c8609,
                    0x1038fb943d774698,
                    0x8a632320fc6e3c35,
                    0x018d3da8846a92bd,
                ])),
                Felt::new(BigInteger384([
                    0x9bdf57a582d79350,
                    0x427cbfd0c36566f2,
                    0x92ce616a9c79463c,
                    0x1542e9001ed3218a,
                    0xfcc6f267e57e840b,
                    0x003f974879b827be,
                ])),
                Felt::new(BigInteger384([
                    0xfaad7918750f1f73,
                    0x4a7a92ab2db52279,
                    0x1c893e8fd74247cf,
                    0x0e81731ba7b9e702,
                    0x5c943a630e57e7d7,
                    0x0137622c0dd68c34,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xe7ccbaf53075dee9,
                    0x9dc2fad14affad14,
                    0x2b401c3073592211,
                    0xc3aa140b42ce5c37,
                    0xfcf7b394dd04920b,
                    0x01898a0cfe15dd0c,
                ])),
                Felt::new(BigInteger384([
                    0xcceeb132df5c3335,
                    0x2392c67cd088abd4,
                    0x38962f5786f40005,
                    0x1685f74c156f2c73,
                    0x003cb58995fad172,
                    0x007b15f0dc90ba87,
                ])),
                Felt::new(BigInteger384([
                    0x01b81da31227ac7a,
                    0x5bc646572f3b29ca,
                    0x59497f8a7ebd7188,
                    0xebacb11d2365080b,
                    0xc3de4d82d4287d19,
                    0x00cbb03375a78900,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x2c4c9bcf41b1a653,
                    0x92a67c37d28ddf7a,
                    0x8217fcb5e121491c,
                    0xbfcaa3c579bca358,
                    0xbe0dd4df54e45373,
                    0x003f5b15e989c7ae,
                ])),
                Felt::new(BigInteger384([
                    0xa25c3a615f500400,
                    0x850245622f4e130f,
                    0x568fb9732faaf071,
                    0x7a95947c22df36a4,
                    0x04188322da4f7c3f,
                    0x0126f96081314dbb,
                ])),
                Felt::new(BigInteger384([
                    0xf6d16923893a5ed0,
                    0x82e339583f70c5f1,
                    0x1e0e6c7253d0b2cd,
                    0x4e742d5919d4c993,
                    0x4c02a11625b084cb,
                    0x004813fa22db41f6,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0x4c36bbb924b90165,
                    0x12cf83e10810067e,
                    0x6acc87d7536df9a1,
                    0xaa163ea2b30ff66a,
                    0x06cb619251c36856,
                    0x01612ba29fa8a973,
                ])),
                Felt::new(BigInteger384([
                    0x36e0132ac7ceea47,
                    0x29f88671d1770db4,
                    0x3e6621fd6c4185c1,
                    0xd21a32097d47dc5b,
                    0x9311b22a4df8c565,
                    0x00c92ebe23e100e1,
                ])),
                Felt::new(BigInteger384([
                    0xe4bd088aaabc6bd3,
                    0x0ebe943994b346ab,
                    0x585bf81be8a9a7d9,
                    0xbacd11694cb82934,
                    0x4006540746c3fffb,
                    0x0049a50313d669ae,
                ])),
            ],
            [
                Felt::new(BigInteger384([
                    0xdfd1a1356db8724d,
                    0x7a5601660f869a4f,
                    0x0f27feed74faf55d,
                    0x463e5f383c31fb1d,
                    0xcac77b33b42c1797,
                    0x018e4ba5d2e96f81,
                ])),
                Felt::new(BigInteger384([
                    0xac9c86461714ce73,
                    0x4ff4768fb862f429,
                    0x8475af490beeb101,
                    0x703f2591b0800ba0,
                    0x30d4336ddafe3e0c,
                    0x009034c887954f9f,
                ])),
                Felt::new(BigInteger384([
                    0x313c2e94ca87ddd8,
                    0x2fe0dca593889a51,
                    0x93157aeef59d968d,
                    0x39ea48b3159c5af5,
                    0xded0e3f96453e0fb,
                    0x012a3c3339c72155,
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
            vec![Felt::zero(); 6],
            vec![Felt::one(); 6],
            vec![
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
                Felt::one(),
                Felt::one(),
                Felt::one(),
            ],
            vec![
                Felt::one(),
                Felt::one(),
                Felt::one(),
                Felt::zero(),
                Felt::zero(),
                Felt::zero(),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x3b050890592c4979,
                    0x0a698c3b95a6d09c,
                    0xe79ebc4d2e4d9516,
                    0x4ef7eabf9d1cd913,
                    0x4399c4f01dc71b99,
                    0x01270bc451c96cf6,
                ])),
                Felt::new(BigInteger384([
                    0xc468a203cfe4f99d,
                    0x5a49518df31b4b3f,
                    0x4747234c22e7ff4b,
                    0xd3619e5187bedff4,
                    0x7c962dab8393656f,
                    0x019f3330fd83fbaf,
                ])),
                Felt::new(BigInteger384([
                    0x07c268b2a9fbd515,
                    0x8a149ef8d4a53387,
                    0xbdb04f318127b006,
                    0x0450850a518e9f7b,
                    0x4ef28f8f00c79d54,
                    0x01093cca98610b3e,
                ])),
                Felt::new(BigInteger384([
                    0xf08da82217d7a4b4,
                    0x1a30eefd2bc37047,
                    0xafeb03dd3db65960,
                    0x19af623b55e34c4f,
                    0xf6cb168d6b7642dc,
                    0x003566a7c4d4e584,
                ])),
                Felt::new(BigInteger384([
                    0x3d5407997f9f083d,
                    0x424df82797dca295,
                    0x96f5d2c32f198de7,
                    0x95fdccfb7f495a2a,
                    0x59448870a5054fb0,
                    0x01a2eb86f9f0057b,
                ])),
                Felt::new(BigInteger384([
                    0xe1fa645257b9e201,
                    0x8c8b1f9a9baafa40,
                    0x55e0068d4cc0dd96,
                    0xa6321b0dc65ff590,
                    0xa2e05ecdef399e05,
                    0x011e8de4e4e39fb7,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xd9fd8b8a52a27952,
                    0x0d2ffe8a38183aaf,
                    0xf5ff1a1b9a03cbd4,
                    0x3d8219bbbff05432,
                    0xf31d225a52ebfba9,
                    0x017c551b23738883,
                ])),
                Felt::new(BigInteger384([
                    0x1d12318fbff804e9,
                    0xf9eaae20feacefc9,
                    0x892e1b00ff0ca753,
                    0x2da1003a0d894b7b,
                    0xddecc8e216a2e09b,
                    0x000c29cba595e73b,
                ])),
                Felt::new(BigInteger384([
                    0x7967a03bfd48c36b,
                    0xd7c5eb3c280b44d9,
                    0x0843e6c72fbe0ddc,
                    0xaee83bddbe87b4f0,
                    0xf5bb1aa8a26cf8d1,
                    0x014f7e08446722d2,
                ])),
                Felt::new(BigInteger384([
                    0x783034a3ec60075a,
                    0x909d895ead39c09b,
                    0x673cc18ea950238d,
                    0xb59890c2c46b112d,
                    0x0a33cdb8a031f4a4,
                    0x0046f5aa90fbc80d,
                ])),
                Felt::new(BigInteger384([
                    0x486197ff87cc1725,
                    0xea5d515cfea54528,
                    0x09ad0ca2faa442d5,
                    0x2fbd6f12f96d863f,
                    0xdf3145af97592e51,
                    0x00763b92a5896e0f,
                ])),
                Felt::new(BigInteger384([
                    0x8f5ff67e884fa4ba,
                    0x4ae577b1fec98802,
                    0xb3ef18dd94db73d9,
                    0x0153a58a22a6f120,
                    0x04949a8ee85d4885,
                    0x009a25bc7136a095,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0xa73a538f696b06cf,
                    0x0d1bed56d440ff98,
                    0x562fcacc943e588c,
                    0xc96a8ae68c6bc05f,
                    0x30b6da84fd8963ec,
                    0x003ec9472fdf31e5,
                ])),
                Felt::new(BigInteger384([
                    0x4076983164704109,
                    0xeeb3a42930cf8749,
                    0xd6943a82557527ae,
                    0xc8f5e5f0798e7608,
                    0x3199607c1ac3ef6a,
                    0x01485274ba0e78e2,
                ])),
                Felt::new(BigInteger384([
                    0x680b47c695531496,
                    0x1d4d4fbb7eb88134,
                    0x2011d86d2fab612a,
                    0xa29437edaafdbf18,
                    0xf5c34d774cf52b37,
                    0x005019b7c0a47610,
                ])),
                Felt::new(BigInteger384([
                    0x93729a89e7ddda4c,
                    0x6776cbfcb9f139d1,
                    0x7cb6dd49973f1f5d,
                    0xf66a048702c761ed,
                    0xc2165e9bd065bb44,
                    0x01897e3789eac5cf,
                ])),
                Felt::new(BigInteger384([
                    0xafc4bc92c17d7037,
                    0x50fb7c52ba52c3a3,
                    0xb2735f991906f209,
                    0xf001f4bd225a93dc,
                    0x1983d9eb70063fb6,
                    0x015b529ea5d88c0f,
                ])),
                Felt::new(BigInteger384([
                    0x115ffd0273a542b4,
                    0x91d2ffb27652bc19,
                    0xa79306d3f8ec985c,
                    0x32ec86f873f3416e,
                    0xb3e199e2088e7e0d,
                    0x0012916a70656b38,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x2922a6ee1a3a99ba,
                    0xbf5f6f40fea7f670,
                    0x97d9ab8a02658ce9,
                    0x4bced580b6b2d92d,
                    0x2e18e180d84c871b,
                    0x012bb337b86f9caa,
                ])),
                Felt::new(BigInteger384([
                    0x7c82518e5d83b652,
                    0xce0df5fb3bd6207d,
                    0x80a64f8a82e4f06d,
                    0x8ef6c98c30567ba8,
                    0x43fa6e692a89f172,
                    0x008529670995a3a6,
                ])),
                Felt::new(BigInteger384([
                    0xdf45926c3f048db9,
                    0xacd9f691e6e9ee1d,
                    0xe99cd27f8b982ab2,
                    0xa9e48564eeca2645,
                    0x41422a3a1f4b9b56,
                    0x0168827072f6e60b,
                ])),
                Felt::new(BigInteger384([
                    0x038d0fdb0c22c730,
                    0xbbbf0d20684b8fef,
                    0xe326caf2176fcc45,
                    0x634cb31bcfb5fb25,
                    0x0bd05008532493d2,
                    0x002bd7a5a0bb5f6d,
                ])),
                Felt::new(BigInteger384([
                    0x02fadded3d8fae93,
                    0x65ccd1ae9b313099,
                    0xe49f415033864035,
                    0x1bb268aeadaa8190,
                    0x2e1fe81cf3037681,
                    0x000fd0b9bd7cb510,
                ])),
                Felt::new(BigInteger384([
                    0x8b168f4d4e17bfef,
                    0x6639e5d8d414fa4e,
                    0x8a81fcfaa09d6815,
                    0xc6a2879de0eea37d,
                    0x711704b4eb0c3bae,
                    0x000cfd84814161be,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x7ec134ff1b6e49a7,
                    0xd339d6bf139a59a3,
                    0x1ad976c2bdf9c3bc,
                    0x221dd3a1537e5e99,
                    0x97b4508852540d9d,
                    0x0175f221976f02a5,
                ])),
                Felt::new(BigInteger384([
                    0x0c2236dc841be241,
                    0x4a034fd56a6ccaf7,
                    0x8a8d42e83b6d7687,
                    0x51a502579a11a584,
                    0x0a21c89d01cb311f,
                    0x01a9ac7b10a7dec6,
                ])),
                Felt::new(BigInteger384([
                    0x4b6357e992498a31,
                    0x8f32eea0769dd81b,
                    0xb9ce74498c2b33ad,
                    0xdd56742257c28933,
                    0x579334f5d75ae43e,
                    0x019dbf6dd8e30991,
                ])),
                Felt::new(BigInteger384([
                    0x94da3da15a170408,
                    0xa348b87bd753035f,
                    0x96344804886bbb4a,
                    0xf1c13a46cae621a3,
                    0xa438fbedd185097e,
                    0x006fcf01fa049f38,
                ])),
                Felt::new(BigInteger384([
                    0x9e2e48b490fbc103,
                    0x43ee337ef3d7fee1,
                    0x0502d2172bf94648,
                    0xd8d6f1a372399d75,
                    0xf289149ae5296196,
                    0x00ab01ad7d2de9bc,
                ])),
                Felt::new(BigInteger384([
                    0xd029f3acd1e6dc7f,
                    0xaef13944184a81b0,
                    0xff31d9ff1fd92f24,
                    0xb143d79106994a6e,
                    0xd5fc1932184c5342,
                    0x01a27ea89dd41350,
                ])),
            ],
            vec![
                Felt::new(BigInteger384([
                    0x74bbd200499718da,
                    0x4bf73ffc0778e24e,
                    0x9fb1fabe0c64e4be,
                    0x97b8a3c66d1e3720,
                    0x0770df926acbbf99,
                    0x017c025be59f725a,
                ])),
                Felt::new(BigInteger384([
                    0xf959eab431e80a62,
                    0x717ad407e9f59047,
                    0x1eb66b0d648dbc52,
                    0xdc4d2df2ca9bab65,
                    0x9da0785be038343a,
                    0x0053b17b7d5e717e,
                ])),
                Felt::new(BigInteger384([
                    0x20eb8066afd9f42c,
                    0xa56eb2dd9990069d,
                    0x1705e3fdb8ae4598,
                    0x9e3a4351f24ad296,
                    0x070d000f02f21be7,
                    0x000a80ccef4b589b,
                ])),
                Felt::new(BigInteger384([
                    0x1ae3cb930945cade,
                    0x872b6a786c015052,
                    0x0f8338e3df278404,
                    0xc76964915d4017ec,
                    0x8200dda689a882dd,
                    0x006aa6e60dae2cae,
                ])),
                Felt::new(BigInteger384([
                    0x64b08d2dbce7e397,
                    0x0c87f267c48784fb,
                    0x59830bd073874c27,
                    0x2359bfb83baa132b,
                    0xb729a47499d3a584,
                    0x013f95ce805e5765,
                ])),
                Felt::new(BigInteger384([
                    0x1c1a48fd0101ba97,
                    0xdb6d6f161081716d,
                    0x545217296e2426b7,
                    0xeabb945b13a508ad,
                    0x7f9c3b1371a03f7c,
                    0x00bd2260ee96c758,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger384([
                0x35cdc96d6230dbd0,
                0xc81e465f54d22546,
                0x4b01c5479de7a212,
                0x7ea1b54afc94829e,
                0x9c0de0a2a161b5f6,
                0x010ad7b576f6b5dc,
            ]))],
            [Felt::new(BigInteger384([
                0x18a225396af1befb,
                0x02c78323b334bd66,
                0x7a66d0843077a1d7,
                0x7c3121c6233ec906,
                0xc3a39b314750dd41,
                0x012d6d9ac5ee0fdf,
            ]))],
            [Felt::new(BigInteger384([
                0x030c77c4f93cf661,
                0xa734a3d74f2dca9c,
                0x91a5695cd6d01072,
                0xc70df3f1e375c333,
                0x90630828671ae7c8,
                0x0010251817c0d202,
            ]))],
            [Felt::new(BigInteger384([
                0x5f8e76b129d58e82,
                0x31a94706403a7b6c,
                0x74b9315c5d7561f5,
                0x97a746ebab176842,
                0x45867aef3a8f2e19,
                0x00c1aca5c8ea031c,
            ]))],
            [Felt::new(BigInteger384([
                0x95830c5a9cea536c,
                0xcdcc9820ef748d5e,
                0x57486839c9bb486f,
                0x089336785213c0b4,
                0x11595797f9fc554b,
                0x0164c0d82d10575a,
            ]))],
            [Felt::new(BigInteger384([
                0x6e2b51ba154a43ec,
                0xb0513fa58e6dd49c,
                0xa2d6838207087c87,
                0x805cec0b22838f72,
                0xed171a5897e526c9,
                0x00f89392031df51a,
            ]))],
            [Felt::new(BigInteger384([
                0xa465f34b2aae7a18,
                0xad72fcef468027c0,
                0x77fb964738b4826d,
                0x2f23bca54ef4b4d6,
                0x47274b1ec281de27,
                0x00020315c512be52,
            ]))],
            [Felt::new(BigInteger384([
                0x63acd8a6ff7fd673,
                0xe5643f8d04c7f99b,
                0x869754fffacefb3a,
                0xda8df054b40ce450,
                0xc272bb7a0f3367d4,
                0x0147f45cd8cd92de,
            ]))],
            [Felt::new(BigInteger384([
                0xe6e6a108ae1d0b7a,
                0x96cc0e5c7d83ad5c,
                0x1ea984696570e11d,
                0x7274b4ba79e25204,
                0x2e8ddfaa73bb5b9f,
                0x01718fddb1627201,
            ]))],
            [Felt::new(BigInteger384([
                0xcf35782e9c0ed076,
                0xc4d05285b373fa83,
                0xf5a7e605f567d292,
                0x077e0794d293567e,
                0x618259ae197a3f68,
                0x00088a8a5f49c9de,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 6));
        }
    }
}
