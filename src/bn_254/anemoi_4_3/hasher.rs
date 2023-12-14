//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::bn_254::anemoi_4_3::AnemoiBn254_4_3;
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiBn254_4_3 {
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
            state[i] += Felt::from_le_bytes_mod_order(&buf[..]);
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiBn254_4_3::permutation(&mut state);
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
            AnemoiBn254_4_3::permutation(&mut state);
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
                AnemoiBn254_4_3::permutation(&mut state);
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
            AnemoiBn254_4_3::permutation(&mut state);
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
        AnemoiBn254_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiBn254_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiBn254_4_3::permutation(&mut state);

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

        let mut state = elems.to_vec();
        AnemoiBn254_4_3::permutation(&mut state);

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

    use super::super::MontFp;
    use super::*;
    use ark_ff::BigInteger;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![MontFp!(
                "20432745219508483737075112941450349001741912103514115650368656955174266846954"
            )],
            vec![
                MontFp!(
                    "17261376915808332770494374489843760193698024999991627486423836476682634792342"
                ),
                MontFp!(
                    "2711365091096434615607465404194778306694577679273248642565922306240490069584"
                ),
            ],
            vec![
                MontFp!(
                    "8289464560404198616204720529986183270774203790697087289442138142574143472869"
                ),
                MontFp!(
                    "6503388940210597606846052244948397415130784671702324184932677516073993902739"
                ),
                MontFp!(
                    "12494012065067620673908783403934459855661412660221220810475843869523620741264"
                ),
            ],
            vec![
                MontFp!(
                    "6100737548852362239629679896700534157387223974728692726528488758501274101508"
                ),
                MontFp!(
                    "1034075118871520129400906140820975138962590924636502420683063869215550352853"
                ),
                MontFp!(
                    "4706344690703076596306077529638812831627888805114341046759839710338341114619"
                ),
                MontFp!(
                    "10265111329293423723499568182059887631968979392874581659769898866652097297682"
                ),
            ],
            vec![
                MontFp!(
                    "4701408353857815496313729819851842611194631679791199093506538029640544031536"
                ),
                MontFp!(
                    "13663484548019227741298309924381099964226562188759124674175702035009748014647"
                ),
                MontFp!(
                    "10903012990401650564805153402004909964121817803312615427905777132208580164620"
                ),
                MontFp!(
                    "892421274168601972538968645411237787288421703409347244650609957682135201725"
                ),
                MontFp!(
                    "14385177450205452897337890139629922150380732738278648170912830047186239612577"
                ),
            ],
            vec![
                MontFp!(
                    "5722493040222900617548603553356800946195646496282547454951565263931506507317"
                ),
                MontFp!(
                    "1454788864692259480300479843107094980368077641242962514027493329047637389467"
                ),
                MontFp!(
                    "15264193593429173857796069771612069399558777396364327797243271484801206161248"
                ),
                MontFp!(
                    "9274323124854978686008589131048534042069269362415258634820132419505646110505"
                ),
                MontFp!(
                    "13199775799620837882691106538712913007861962754860864416481061262930737763648"
                ),
                MontFp!(
                    "10608842350562139863360914040076045254737689477476985326192728192926487687353"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "19703523448579527760453009550979032245776004803110801804184952516345545484750"
            )],
            [MontFp!(
                "12973666487717614852762188211803523490095093031801540563900817594202005308802"
            )],
            [MontFp!(
                "18284813920150210681819283200849662485270180033653572977537341551414801681092"
            )],
            [MontFp!(
                "20462747784775344897016485483699615299032553273492896135582323436575817639745"
            )],
            [MontFp!(
                "10781954120022946129198657206030233388851809649708827816000100686642826756186"
            )],
            [MontFp!(
                "12526505721863231589795362666433783278187220390157886071072147086701096303769"
            )],
            [MontFp!(
                "7009524382317632623954448624714051899964168437588222778606975740262233707215"
            )],
            [MontFp!(
                "2966375203485447070554991402557380328559683466747434335703609492669814895298"
            )],
            [MontFp!(
                "12088733942715502991745350926225429883284691427506173564600971345401987538123"
            )],
            [MontFp!(
                "9118037005619176786566133890523543800260749887194227256358900285590502031619"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiBn254_4_3::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "19703523448579527760453009550979032245776004803110801804184952516345545484750"
            )],
            [MontFp!(
                "12973666487717614852762188211803523490095093031801540563900817594202005308802"
            )],
            [MontFp!(
                "18284813920150210681819283200849662485270180033653572977537341551414801681092"
            )],
            [MontFp!(
                "20462747784775344897016485483699615299032553273492896135582323436575817639745"
            )],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&input[0].into_bigint().to_bytes_le()[0..31]);
            bytes[31..62].copy_from_slice(&input[1].into_bigint().to_bytes_le()[0..31]);
            bytes[62..93].copy_from_slice(&input[2].into_bigint().to_bytes_le()[0..31]);
            bytes[93..124].copy_from_slice(&input[3].into_bigint().to_bytes_le()[0..31]);

            assert_eq!(expected, AnemoiBn254_4_3::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [
                MontFp!(
                    "12579583631457685948757922530037597275799132277960261798393234116995940690573"
                ),
                MontFp!(
                    "4623603763873116191261598634915842923449429138501854205009131372188500382569"
                ),
            ],
            [
                MontFp!(
                    "5249097963515294866812168766214803597450406939529597566853236998246729301177"
                ),
                MontFp!(
                    "15492516849241804090050703963635961560538949991737890922483621920781926565426"
                ),
            ],
            [
                MontFp!(
                    "16262314304864052916359262342765773764407063612068095818639130065173528197022"
                ),
                MontFp!(
                    "12598181112269515226794235411820050069208473764596273084730839643497146520604"
                ),
            ],
            [
                MontFp!(
                    "1798763111022460126883843032453841834387052919381658240033634438924107153712"
                ),
                MontFp!(
                    "21469440594109799707805169083974578735965792209510796507491380210197386329620"
                ),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "17203187395330802140019521164953440199248561416462116003402365489184441073142"
            )],
            [MontFp!(
                "20741614812757098956862872729850765157989356931267488489336858919028655866603"
            )],
            [MontFp!(
                "6972252545294292920907092009328548744919226219366545240680931814025448509043"
            )],
            [MontFp!(
                "1379960833292984612442606371171145481656533971594631084835976754476267274749"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_4_3::compress_k(input, 4));
        }
    }
}
