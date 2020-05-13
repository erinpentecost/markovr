extern crate markovr;

// The training data.
const BOXES: Vec<Vec<string>> = "                                
 ┏━━━━┳━━━━━━┓ ┏━┳━━┳━━━━━━━━━━┓ 
 ┃    ┃ ┏━┓  ┃ ┃ ┃  ┃          ┃ 
 ┣━━━━╋━╋━╋━━╋━┫ ┃ ┏╋━━━━┓     ┃ 
 ┃    ┃ ┗━┛  ┃ ┃ ┃ ┗╋━━━━┛     ┃ 
 ┗━━━━┻━━━━━━┛ ┗━┻━━┻━━━━━━━━━━┛ 
                                 
 "
.lines()
.map(c => c.chars().collect())
.collect();

pub fn main() {
    // Create a new, third-order Markov Chain.
    let mut m = markovr::MarkovChain::new(3);

    // Train the model.
    for r, row in (1..(BOXES.len())).enumerate() {
        for c, result in (1..(row.len())).enumerate() {
            // Build up a view of the neighbors.
            let neighbors = &[BOXES[r-1][c-1], BOXES[r-1][c], BOXES[r][c-1]];
            m.train(neighbors, result, 1);
    }

    // Generate values from the model.
    let mut rand_map : Vec<Vec<char>> = vec!(vec!(&['┏', '━']),vec!(&['┃']));
    let mut last: Option<char> = Some('a');
    while last.is_some() {
        print!("{} ", last.unwrap());
        last = m.generate(&[last.unwrap()]);
    }
    // Prints: a b c d e f g h i j k l m n o p q r s t u v w x y z

    // What's the probability that 'z' follows 'y'?
    print!("\n{}", m.probability(&['y'], 'z'));
    // Prints: 1
    // What's the probability that 'z' follows 'a'?
    print!("\n{}\n", m.probability(&['a'], 'z'));
    // Prints: 0
}
