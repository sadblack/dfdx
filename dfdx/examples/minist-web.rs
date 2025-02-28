

use std::{arch::x86_64, net::SocketAddr, path::Path, sync::Arc, time::Instant};
use axum::{debug_handler, extract::State, http::{HeaderValue, StatusCode}, response::IntoResponse, routing::post, Json, Router};
use indicatif::ProgressIterator;
use mnist::*;
use rand::prelude::{SeedableRng, StdRng};

use dfdx::{data::*, prelude::*};
use serde::Deserialize;
use tower_http::cors::{Any, CorsLayer};
// our network structure
type Mlp = (
    (LinearConstConfig<784, 512>, ReLU),
    (LinearConstConfig<512, 128>, ReLU),
    (LinearConstConfig<128, 32>, ReLU),
    LinearConstConfig<32, 10>,
);

// ANSI 转义序列：蓝色前景色
const BLUE: &str = "\x1B[34m";
// ANSI 转义序列：重置颜色
const RESET: &str = "\x1B[0m";

#[debug_handler]
// 处理 POST 请求的函数
async fn random_array(
    State(state): 
    State<
        Arc<
            (
                (Linear<Const<784>, Const<512>, f32, Cpu>, ReLU), (Linear<Const<512>, Const<128>, f32, Cpu>, ReLU), 
                (Linear<Const<128>, Const<32>, f32, Cpu>, ReLU), 
                Linear<Const<32>, Const<10>, f32, Cpu>
            )
        >
    >,
    Json(input): Json<Vec<f32>>
) -> impl IntoResponse {

    if input.len() != 784 {
        return Err((StatusCode::BAD_REQUEST, "Input array must have length 784".to_string()));
    }

    //input 是一个 784个 元素 的数组，遍历这个数组，每 28个打印一行，如果值 > 0.001, 打印 1，如果值为 0， 打印 0
    for (i, val) in input.iter().enumerate() {
        // let print_char = if *val > 0.001 { '1' } else { '0' };
        // print!("{} ", print_char);

        let print_char = if *val > 0.00001 {
            format!("{}{}{}", BLUE, '1', RESET)
        } else {
            "0".to_string()
        };
        print!(" {} ", print_char);

        if (i + 1) % 28 == 0 {
            println!();
        }
    }
    println!();
    println!();
    println!();

    let req: Tensor<Rank1<784>, f32, _> = AutoDevice::default().tensor(input);

    let r: [f32; 10] = state.forward(req).softmax().array();

    Ok((StatusCode::OK, Json(adjust_softmax_output(r))))
}


#[tokio::main]
async fn main() {

    let cors = CorsLayer::permissive();

    let model = load_model();

    let model: 
    Arc<
        (
            (Linear<Const<784>, Const<512>, f32, Cpu>, ReLU), (Linear<Const<512>, Const<128>, f32, Cpu>, ReLU), 
            (Linear<Const<128>, Const<32>, f32, Cpu>, ReLU), 
            Linear<Const<32>, Const<10>, f32, Cpu>
        )
    > 
    = Arc::new(model);

    let app = Router::new()
    .route("/random_array", post(random_array))
    .with_state(model)
    .layer(cors);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();

}

fn adjust_softmax_output(mut arr: [f32; 10]) -> [f32; 10] {


    // println!("result array: {:?}", arr);


    // 第一步：将数组中的每个元素保留两位小数
    for i in 0..arr.len() {
        arr[i] = (arr[i] * 100.0).round() / 100.0;
    }

    // 第二步：计算保留两位小数后数组元素的总和
    let mut sum: f32 = arr.iter().sum();

    // 第三步：找出数组中绝对值最大的元素的索引
    let mut max_index = 0;
    let mut max_abs = arr[0].abs();
    for i in 1..arr.len() {
        if arr[i].abs() > max_abs {
            max_abs = arr[i].abs();
            max_index = i;
        }
    }

    // 第四步：调整绝对值最大的元素，使得总和为 1
    arr[max_index] += 1.0 - sum;

    // 第五步：再次确保调整后的元素仍然保留两位小数
    arr[max_index] = (arr[max_index] * 100.0).round() / 100.0;

    arr
}


fn load_model() -> ((Linear<Const<784>, Const<512>, f32, Cpu>, ReLU), (Linear<Const<512>, Const<128>, f32, Cpu>, ReLU), (Linear<Const<128>, Const<32>, f32, Cpu>, ReLU), Linear<Const<32>, Const<10>, f32, Cpu>) {
    
    // ftz substantially improves performance
    dfdx::flush_denormals_to_zero();
    let exe_path = std::env::current_exe().expect("Failed to get executable path");
    println!("Executable path: {:?}", exe_path);
    let mnist_model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let exe_dir = exe_path.parent().expect("Failed to get executable directory");
        // 向上找三层目录
        let base_dir = exe_dir
            .parent()
            .and_then(|d| d.parent())
            .and_then(|d| d.parent())
            .expect("Failed to find project root directory");
        
        let dataset_path = base_dir;
        
        // 检查路径是否存在
        if !dataset_path.exists() {
            println!("Warning: Dataset path does not exist: {:?}", dataset_path);
            println!("Please ensure the MNIST dataset is downloaded to the correct location");
        }
        
        dataset_path.to_str().expect("Invalid path").to_string()
    });

    println!("Loading mnist model from path: {mnist_model_path}");

    let dev = AutoDevice::default();


    // initialize model, gradients, and optimizer
    // let mut model = dev.build_module::<f32>(Mlp::default());
    // let mut grads = model.alloc_grads();
    // let mut opt = dfdx::nn::optim::Adam::new(&model, Default::default());
    let mut model = dev.build_module::<f32>(Mlp::default());
    model
        .load_safetensors(mnist_model_path + "//mnist.safetensors")
        .expect("Failed to load model");

        model
}












    /*
        1. 怎么把图片转为矩阵
        2. 为什么转为 32 * 784 的矩阵。32 的意思是一次处理 32张，每行代表一张图片。使用的时候，只需保证 行是784就可以，如果输入列是1，输出的列也是 1
        3. 优化策略是什么
        4. 为什么选这些层
        5. loss 怎么算的
        6. 输出的是矩阵，怎么判断是哪个数字
        7. 保存的权重，怎么用

        我训练得到了一个mnist的权重文件，名字叫 test.safetensors，1m多，想在浏览器里运行。我想写一个页面, 可以让用户画一个数字，然后识别出来。
        这个页面分为三个部分，
        左边一个是画板，用户可以按住鼠标左键，在上面画出数字，画板下面是一个清除按钮
        中间是画板上的内容转换成的图片，是一个 28 * 28 的图片
        右边是识别结果。是一个柱状图，显示0-9的概率。因为模型返回的是一个数组，长度是10

        每秒采样一次画板，把画板上的内容转换成图片，然后把图片转换成矩阵，然后把矩阵输入模型，得到结果，然后把结果显示在右边的柱状图上。
        请给出这个前端页面的代码
     */
