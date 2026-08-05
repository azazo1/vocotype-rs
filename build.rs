use std::env;
use std::path::Path;
use std::process::Command;

fn run_git(args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .env("LC_ALL", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
}

fn parse_describe(describe: &str) -> Option<(String, u64, String)> {
    let hash_start = describe.rfind("-g")?;
    let hash = &describe[hash_start + 2..];
    if hash.len() < 6 || !hash.chars().all(|character| character.is_ascii_hexdigit()) {
        return None;
    }

    let before = &describe[..hash_start];
    let count_dash = before.rfind('-')?;
    let count = before[count_dash + 1..].parse().ok()?;
    Some((before[..count_dash].to_string(), count, hash.to_string()))
}

fn git_commit_version(package_version: &str) -> Option<String> {
    let describe = run_git(&[
        "describe",
        "--tags",
        "--match",
        "v*",
        "--always",
        "--long",
        "--dirty",
        "--abbrev=6",
    ])?;
    let short_hash = run_git(&["rev-parse", "--short=6", "HEAD"])?;
    let dirty = describe.ends_with("-dirty");
    let clean = describe.strip_suffix("-dirty").unwrap_or(&describe);

    if let Some((tag, commits, _)) = parse_describe(clean) {
        if dirty {
            return Some(format!("{tag}^{short_hash}"));
        }
        if commits == 0 {
            return Some(tag);
        }
        return Some(format!("{tag}-{short_hash}"));
    }

    if dirty {
        Some(format!("v{package_version}^{short_hash}"))
    } else {
        Some(format!("v{package_version}-{short_hash}"))
    }
}

fn main() {
    let package_version =
        env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    let build_version =
        git_commit_version(&package_version).unwrap_or_else(|| format!("v{package_version}"));

    println!("cargo:rustc-env=VOCOTYPE_BUILD_VERSION={build_version}");
    println!("cargo:rerun-if-changed=build.rs");
    for git_file in [".git/HEAD", ".git/index", ".git/packed-refs"] {
        if Path::new(git_file).exists() {
            println!("cargo:rerun-if-changed={git_file}");
        }
    }
}
