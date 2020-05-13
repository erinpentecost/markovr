extern crate markovr;

pub fn main() {
    // Create a new, fourth-order Markov Chain.
    // We'll keep track of each orthogonal neighbor,
    // and allow for any one of them to be unknown.
    let mut m = markovr::MarkovChain::<char>::new(4, &[0, 1, 2, 3]);

    let train: Vec<Vec<char>> = "                                
    ┏━━━━┳━━━━━━┓ ┏━┳━━┳━━━━━━━━━━┓ 
    ┃    ┃ ┏━┓  ┃ ┃ ┃  ┃          ┃ 
    ┣━━━━╋━╋━╋━━╋━┫ ┃ ┏╋━━━━┓     ┃ 
    ┃    ┃ ┗━┛  ┃ ┃ ┃ ┗╋━━━━┛     ┃ 
    ┗━━━━┻━━━━━━┛ ┗━┻━━┻━━━━━━━━━━┛ 
                                    
    "
    .lines()
    .map(|c| c.chars().collect())
    .collect();

    // Train the model.
    for r in 1..(train.len()) {
        let ref row = train[r];
        for c in 1..(row.len()) {
            // Build up a view of the neighbors.
            let neighbors = &[
                train[r - 1][c],
                train[r][c - 1],
                train[r][c + 1],
                train[r - 1][c],
            ];
            m.train(neighbors, train[r][c], 1);
        }
    }

    // Generate values from the model.
    let mut map: [[Option<char>; 16]; 16] = [[None; 16]; 16];
    //let mut rand_map : Vec<Vec<char>> = vec!(vec!(&['┏', '━']),vec!(&['┃']));
    for r in 0..16 {
        for c in 0..16 {
            let neighbors = &[map[r - 1][c], map[r][c - 1], map[r][c + 1], map[r - 1][c]];

            map[r][c] = m.generate_from_partial(neighbors);
            match map[r][c] {
                Some(c) => print!("{}", c),
                None => print!("?"),
            }
        }
        print!("\n");
    }
    // Prints: a b c d e f g h i j k l m n o p q r s t u v w x y z
}
