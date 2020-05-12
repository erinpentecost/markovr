extern crate markovr;

pub fn main() {
    // Create a new, first-order Markov Chain.
    let mut m = markovr::MarkovChain::new(1);

    // alpha will be both our encoding mapping and training data.
    // markovr only speaks u64s, so the indices of alpha will be the encoding.
    let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    // Train the model.
    for i in 1..alpha.len() {
        m.train(&[(i - 1) as u64], i as u64, 1);
    }

    // Generate values from the model.
    let mut last: Option<u64> = Some('a');
    while last.is_some() {
        print!("{} ", last.unwrap());
        // encode the character
        let encoded_last: u64 = alpha
            .iter()
            .enumerate()
            .filter(|(_i, v)| **v == last.unwrap())
            .next()
            .unwrap()
            .0 as u64;
        let encoded_next = m.generate(&[encoded_last]);
        match encoded_next {
            Some(v) => {
                last = Some(alpha[v as usize]);
            }
            None => {
                last = None;
            }
        }
    }
}
// Prints: a b c d e f g h i j k l m n o p q r s t u v w x y z
