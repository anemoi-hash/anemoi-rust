//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiVesta_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;
use ark_ff::PrimeField;

use super::Felt;
use super::{One, Zero};

impl Sponge<Felt> for AnemoiVesta_4_3 {
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
                AnemoiVesta_4_3::permutation(&mut state);
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
            AnemoiVesta_4_3::permutation(&mut state);
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
                AnemoiVesta_4_3::permutation(&mut state);
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
            AnemoiVesta_4_3::permutation(&mut state);
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
        AnemoiVesta_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiVesta_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.to_vec();
        AnemoiVesta_4_3::permutation(&mut state);

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
        AnemoiVesta_4_3::permutation(&mut state);

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
                "7764516216146978186543732804836859542970724356478521491256477355492627411676"
            )],
            vec![
                MontFp!(
                    "22907694841645630585206426872085849258295842985677302589412328525870756066394"
                ),
                MontFp!(
                    "4169673177469938832983408323556363511312758627908452193980748118915671330837"
                ),
            ],
            vec![
                MontFp!(
                    "15435395061201999509908033209911731823908765952074391516984626065576002144425"
                ),
                MontFp!(
                    "1373703704887686828762179144173833609201564828560942020455143134110608444233"
                ),
                MontFp!(
                    "20363545585638743235928574358837257410927557124964422516999296287451458583899"
                ),
            ],
            vec![
                MontFp!(
                    "19737755197119505596565023771317041889892926723277350338870313434841836325558"
                ),
                MontFp!(
                    "23729531450816876722219741944080547049408566868054130749873492469628373649772"
                ),
                MontFp!(
                    "26204513837294252303065523757180069529491792275684413266576931716040028415701"
                ),
                MontFp!(
                    "5497113521591738232212478403300478363119804014193035879001932507184179361880"
                ),
            ],
            vec![
                MontFp!(
                    "13375370216302311521700013725347609399532334479246690464800191396923778241218"
                ),
                MontFp!(
                    "14876027157593517139229423836996423271630362084655756694521458791463669911260"
                ),
                MontFp!(
                    "16702498949380500451475191924085103265198600790916501051205760468419176003508"
                ),
                MontFp!(
                    "6695341102210164665179608171900587634238989653216410510668285866345844659563"
                ),
                MontFp!(
                    "17339489542647227971818078732159861692512722239006820520684287465025064770535"
                ),
            ],
            vec![
                MontFp!(
                    "18371458419042225327196408930995066412765059814499030652366683114741124920534"
                ),
                MontFp!(
                    "26104863697817533244632757166247186677246954963079946787997558864637706562787"
                ),
                MontFp!(
                    "12059940180902035024688447118770018806865296733534891123984848371927300652705"
                ),
                MontFp!(
                    "176760369206591843207298018372745619374153399069720199050588194776740716420"
                ),
                MontFp!(
                    "7879750185791152904736606040999994226397391884337383165443171129026430613343"
                ),
                MontFp!(
                    "21922030067662672999471329324457178633313324829063788722213944047268018712948"
                ),
            ],
        ];

        let output_data = [
            [MontFp!(
                "18477755575328750816885850798387952933259583428511618735955991488059329852802"
            )],
            [MontFp!(
                "1089622155824349342136353788060036422274330799810677835411793924529134549007"
            )],
            [MontFp!(
                "23383407270625247748646649290487223541381602019790071288068181519850316119688"
            )],
            [MontFp!(
                "15333143339905011434961662026430877746311759859432025806436665255558283094312"
            )],
            [MontFp!(
                "7957074743442196816441071837512213774788799178524178087337708054612561105779"
            )],
            [MontFp!(
                "27964596164958766866855168409189386623161234288265058577280237576522926353764"
            )],
            [MontFp!(
                "8832233859471293253974233791949469443578850234406402147920527938156237262199"
            )],
            [MontFp!(
                "15732110983354407176919337835090149963515384873765009451440597278637567592368"
            )],
            [MontFp!(
                "10330960395688575039826453108111489685452424322682532413106886720241532044898"
            )],
            [MontFp!(
                "28601285983493688442854476172888368799474301316045325162155888974354765051381"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiVesta_4_3::hash_field(input).to_elements());
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
                "18477755575328750816885850798387952933259583428511618735955991488059329852802"
            )],
            [MontFp!(
                "1089622155824349342136353788060036422274330799810677835411793924529134549007"
            )],
            [MontFp!(
                "23383407270625247748646649290487223541381602019790071288068181519850316119688"
            )],
            [MontFp!(
                "15333143339905011434961662026430877746311759859432025806436665255558283094312"
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

            assert_eq!(expected, AnemoiVesta_4_3::hash(&bytes).to_elements());
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
                    "17814074569133619376001583724909369837639006754012602478069266346597252318717"
                ),
                MontFp!(
                    "4645324191354213145361406141970260204294326396494284352043313444215651448956"
                ),
            ],
            [
                MontFp!(
                    "24893920820224764770810638096592977772418417663317024760390036326700338198854"
                ),
                MontFp!(
                    "15860435497577179224758206597683167799910420450474978252299280484428162460455"
                ),
            ],
            [
                MontFp!(
                    "27126766676817950878771190712975642236437683248501595320269350109633958938314"
                ),
                MontFp!(
                    "14615425120361368767589882074890977200866828715767578149241602916394899244813"
                ),
            ],
            [
                MontFp!(
                    "16280243636991791422017348691987967083568430057992986467671369808716501319725"
                ),
                MontFp!(
                    "12714680685026350363019468164077713565648590311659840387446374044983187263140"
                ),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [MontFp!(
                "22459398760487832521362989866879630041933333150506886830112579790812903767673"
            )],
            [MontFp!(
                "11806334008472895139676098442104168608965781631850355633009574062735137711212"
            )],
            [MontFp!(
                "12794169487850270790468326535694642473941455482327526089831210277635495235030"
            )],
            [MontFp!(
                "46902012689092929144070603893703685853963887711179475438001105306325634768"
            )],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress_k(input, 4));
        }
    }
}
