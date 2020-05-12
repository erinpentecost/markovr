# markovr [![crates.io](https://img.shields.io/crates/v/markovr.svg)](https://crates.io/crates/markovr)

**Higher-order Markov Chains** can have longer memories than your [typical Markov Chain](https://en.wikipedia.org/wiki/Markov_chain), which looks back only 1 element. They are the basic building block for the [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse) algorithm. A zeroth-order Markov Chain is the equivalent of a weighted die.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
markovr = {version = "0.2"}
```

Alternatively, if you don't want to bring in the [rand](https://crates.io/crates/rand) crate into your dependency tree:

```toml
[dependencies]
markovr = {version = "0.2", features = []}
```

And then, in your program:

```rust
extern crate markovr;

pub fn main() {
    // Create a new, first-order Markov Chain.
    let mut m = markovr::MarkovChain::new(1);

    // alpha will be our training data.
    let alpha: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();

    // Train the model.
    for i in 1..alpha.len() {
        m.train(&[alpha[i - 1]], alpha[i], 1);
    }

    // Generate values from the model.
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
```
