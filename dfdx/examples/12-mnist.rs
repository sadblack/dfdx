//! This example ties all the previous ones together
//! to build a neural network that learns to recognize
//! the MNIST digits.
//!
//! To download the MNIST dataset, do the following:
//! ```
//! mkdir tmp/ && cd tmp/
//! curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz \
//!     -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
//! gunzip t*-ubyte.gz
//! cd -
//! ```
//! Then, you may run this example with:
//! ```
//! cargo run --example 12-mnist -- tmp/
//! ```

use std::{path::Path, time::Instant};
use indicatif::ProgressIterator;
use mnist::*;
use rand::prelude::{SeedableRng, StdRng};

use dfdx::{data::*, prelude::*};

struct MnistTrainSet(Mnist);

impl MnistTrainSet {
    fn new(path: &str) -> Self {
        Self(MnistBuilder::new().base_path(path).finalize())
    }
}

impl ExactSizeDataset for MnistTrainSet {
    type Item<'a> = (Vec<f32>, usize) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        let mut img_data: Vec<f32> = Vec::with_capacity(784);
        let start = 784 * index;
        img_data.extend(
            self.0.trn_img[start..start + 784]
                .iter()
                .map(|x| *x as f32 / 255.0),
        );
        (img_data, self.0.trn_lbl[index] as usize)
    }
    fn len(&self) -> usize {
        self.0.trn_lbl.len()
    }
}

// our network structure
type Mlp = (
    (LinearConstConfig<784, 512>, ReLU),
    (LinearConstConfig<512, 128>, ReLU),
    (LinearConstConfig<128, 32>, ReLU),
    LinearConstConfig<32, 10>,
);

// training batch size
const BATCH_SIZE: usize = 32;

fn main() {
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();
    let exe_path = std::env::current_exe().expect("Failed to get executable path");
    println!("Executable path: {:?}", exe_path);
    let mnist_path = std::env::args().nth(1).unwrap_or_else(|| {
        let exe_dir = exe_path.parent().expect("Failed to get executable directory");
        // 向上找三层目录
        let base_dir = exe_dir
            .parent()
            .and_then(|d| d.parent())
            .and_then(|d| d.parent())
            .expect("Failed to find project root directory");
        
        let dataset_path = base_dir.join("datasets").join("MNIST").join("raw");
        
        // 检查路径是否存在
        if !dataset_path.exists() {
            println!("Warning: Dataset path does not exist: {:?}", dataset_path);
            println!("Please ensure the MNIST dataset is downloaded to the correct location");
        }
        
        dataset_path.to_str().expect("Invalid path").to_string()
    });

    println!("Loading mnist from args[1] = {mnist_path}");
    println!("Override mnist path with `cargo run --example 12-mnist -- <path to mnist>`");

    let dev = Device::default();
    let mut rng = StdRng::seed_from_u64(0);

    // initialize model, gradients, and optimizer
    let mut model = dev.build_module::<f32>(Mlp::default());
    let mut grads = model.alloc_grads();
    let mut opt = dfdx::nn::optim::Adam::new(&model, Default::default());

    // initialize dataset
    let dataset = MnistTrainSet::new(&mnist_path);
    println!("Found {:?} training images", dataset.len());

    let preprocess = |(img, lbl): <MnistTrainSet as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[lbl] = 1.0;
        (
            dev.tensor_from_vec(img, (Const::<784>,)),
            dev.tensor(one_hotted),
        )
    };

    for i_epoch in 0..500 {
        let mut total_epoch_loss = 0.0;
        let mut num_batches = 0;
        let start = Instant::now();
        for (img, lbl) in dataset
            .shuffled(&mut rng)
            .map(preprocess)
            .batch_exact(Const::<BATCH_SIZE>)
            .collate()
            .stack()
            .progress()
        {
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);

            total_epoch_loss += loss.array();
            num_batches += 1;

            grads = loss.backward();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
        }
        let dur = Instant::now() - start;

        let loss_num = BATCH_SIZE as f32 * total_epoch_loss / num_batches as f32;

        println!(
            "Epoch {i_epoch} in {:?} ({:.3} batches/s): avg sample loss {:.5}",
            dur,
            num_batches as f32 / dur.as_secs_f32(),
            loss_num,
        );
    }
    //保存权重
    #[cfg(feature = "safetensors")]
    model
        .save_safetensors("13-mnist.safetensors")
        .expect("failed to save model");
}

    /*
        1. 怎么把图片转为矩阵
        2. 为什么转为 32 * 784 的矩阵。32 的意思是一次处理 32张，每行代表一张图片。使用的时候，只需保证 行是784就可以，如果输入列是1，输出的列也是 1
        3. 优化策略是什么
        4. 为什么选这些层
        5. loss 怎么算的，意味着什么
        6. batch 有什么用


     */
