//! Implementation of the Anemoi permutation

use super::{sbox, Felt, MontFp};
use crate::{Anemoi, Jive, Sponge};
use ark_ff::{One, Zero};
/// Digest for Anemoi
mod digest;
/// Sponge for Anemoi
mod hasher;
/// Round constants for Anemoi
mod round_constants;

pub use digest::AnemoiDigest;

// ANEMOI CONSTANTS
// ================================================================================================

/// Function state is set to 2 field elements or 64 bytes.
/// 1 element of the state is reserved for capacity.
pub const STATE_WIDTH: usize = 2;
/// 1 element of the state is reserved for rate.
pub const RATE_WIDTH: usize = 1;

/// The state is divided into two even-length rows.
pub const NUM_COLUMNS: usize = 1;

/// One element (32-bytes) is returned as digest.
pub const DIGEST_SIZE: usize = RATE_WIDTH;

/// The number of rounds is set to 21 to provide 128-bit security level.
pub const NUM_HASH_ROUNDS: usize = 21;

// ANEMOI INSTANTIATION
// ================================================================================================

/// An Anemoi instantiation over Vesta basefield with 1 column and rate 1.
#[derive(Debug, Clone)]
pub struct AnemoiVesta_2_1;

impl<'a> Anemoi<'a, Felt> for AnemoiVesta_2_1 {
    const NUM_COLUMNS: usize = NUM_COLUMNS;
    const NUM_ROUNDS: usize = NUM_HASH_ROUNDS;

    const WIDTH: usize = STATE_WIDTH;
    const RATE: usize = RATE_WIDTH;
    const OUTPUT_SIZE: usize = DIGEST_SIZE;

    const ARK_C: &'a [Felt] = &round_constants::C;
    const ARK_D: &'a [Felt] = &round_constants::D;

    const GROUP_GENERATOR: u32 = sbox::BETA;

    const ALPHA: u32 = sbox::ALPHA;
    const INV_ALPHA: Felt = sbox::INV_ALPHA;
    const BETA: u32 = sbox::BETA;
    const DELTA: Felt = sbox::DELTA;

    fn exp_by_inv_alpha(x: Felt) -> Felt {
        sbox::exp_by_inv_alpha(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbox() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let mut input = [
            [Felt::zero(), Felt::zero()],
            [Felt::one(), Felt::one()],
            [Felt::zero(), Felt::one()],
            [Felt::one(), Felt::zero()],
            [
                MontFp!(
                    "10403685622187338496676844159192081731060984895047267868970314023677138575474"
                ),
                MontFp!(
                    "26692143025703589755822959174689724204213429494558130968834356051832228783321"
                ),
            ],
            [
                MontFp!(
                    "11083269342651266673921643458883891120042506350625929791622832610969208196607"
                ),
                MontFp!(
                    "1953573848928623793843704414666375386174461809928685924317395735306133465181"
                ),
            ],
            [
                MontFp!(
                    "15301123091319757024370296695504172236894218806603968176307001943856977747722"
                ),
                MontFp!(
                    "18494761567018646279105628665220627411445922785270663034324810380054915599812"
                ),
            ],
            [
                MontFp!(
                    "11016756002634006512514914914257673803148200338799253823972231973739990986302"
                ),
                MontFp!(
                    "6587924741283998615064635959232428707221544326700215326966307885941351335897"
                ),
            ],
            [
                MontFp!(
                    "1989621086890026999537167291360304582470029001443043660249560541178644694627"
                ),
                MontFp!(
                    "2201628511959366116857902819637154838057028847501245429346761337311313529725"
                ),
            ],
            [
                MontFp!(
                    "17001537352177538606843893918074436419403991546928777655566016185819020620076"
                ),
                MontFp!(
                    "21769394458064296524596890686754956680992668837455600264453508465453114300499"
                ),
            ],
        ];

        let output = [
            [
                MontFp!(
                    "11579208923731619542357098500868790785345222592776658951871897099357345179239"
                ),
                Felt::zero(),
            ],
            [
                MontFp!(
                    "13565375592455225805458964934459476225912655788084948267498268443578124721632"
                ),
                MontFp!(
                    "9688406656496048098325282220348971925838074278218514686842913989361614061362"
                ),
            ],
            [
                MontFp!(
                    "2367797382831619836622158180640631392193461256316785256748737102603438284997"
                ),
                MontFp!(
                    "22890698294176523999447614696141668677027690702028879487883356180097137464994"
                ),
            ],
            [
                MontFp!(
                    "11579208923731619542357098500868790785345222592776658951871897099357345179245"
                ),
                MontFp!(
                    "28948022309329048855892746252171976963363056481941647379679742748393362948096"
                ),
            ],
            [
                MontFp!(
                    "7229165017443906767622585884374304684130765951831734347430899601571384172106"
                ),
                MontFp!(
                    "17186571240634805451677777319828211309724314161534779729178855075911507518178"
                ),
            ],
            [
                MontFp!(
                    "22094751406364599327979467485502691604741323090648163195126472214015255051094"
                ),
                MontFp!(
                    "24672861152559669484078819529785630354683262341488703570100793223767652751073"
                ),
            ],
            [
                MontFp!(
                    "18153986297592575843739235907025865820788018110394974879950689484566885917283"
                ),
                MontFp!(
                    "8761887547072102502715179527981546024180136743223516992337525037544415823723"
                ),
            ],
            [
                MontFp!(
                    "6504823060778863709590368534937957138596774245307615204135830506019587424463"
                ),
                MontFp!(
                    "17360335234679575475171190819808925556523201605131000602748421173274342922620"
                ),
            ],
            [
                MontFp!(
                    "2861027999303393362309602332662059778984051102542167942577428341133728259315"
                ),
                MontFp!(
                    "7633524261411210393878411081927866791992057233891693448302180714099458275436"
                ),
            ],
            [
                MontFp!(
                    "24047861915305283538782525601995635272644343236727145730243967615034082789254"
                ),
                MontFp!(
                    "13139339885763663718126681299471742859020269584204894915009934034197139502359"
                ),
            ],
        ];

        for i in input.iter_mut() {
            AnemoiVesta_2_1::sbox_layer(i);
        }

        for (&i, o) in input.iter().zip(output) {
            assert_eq!(i, o);
        }
    }
}
