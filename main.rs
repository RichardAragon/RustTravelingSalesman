use std::time::{SystemTime, UNIX_EPOCH};

const NUM_CITIES: usize = 20;
const NUM_PARTICLES: usize = 500;
const MAX_ITERATIONS: usize = 2000;
const INITIAL_INERTIA_WEIGHT: f64 = 0.9;
const FINAL_INERTIA_WEIGHT: f64 = 0.4;
const COGNITIVE_COMPONENT: f64 = 1.49445;
const SOCIAL_COMPONENT: f64 = 1.49445;
const MUTATION_RATE: f64 = 0.1;
const PRUNE_PERCENTAGE: usize = 10;

#[derive(Clone, Copy)]
struct City {
    x: i32,
    y: i32,
}

#[derive(Clone)]
struct Particle {
    position: Vec<usize>,
    best_position: Vec<usize>,
    best_cost: f64,
    cost: f64,
}

impl Particle {
    fn new(position: Vec<usize>, cost: f64) -> Self {
        Particle {
            best_position: position.clone(),
            best_cost: cost,
            position,
            cost,
        }
    }
}

fn generate_cities() -> Vec<City> {
    let mut cities = Vec::with_capacity(NUM_CITIES);
    for _ in 0..NUM_CITIES {
        cities.push(City {
            x: random_range(0, 100),
            y: random_range(0, 100),
        });
    }
    cities
}

fn random_range(min: i32, max: i32) -> i32 {
    let epoch_time = SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_nanos();
    let seed = (epoch_time % (max - min) as u128) as i32;
    min + seed
}

fn shuffle_vec(vec: &mut Vec<usize>) {
    let len = vec.len();
    for i in 0..len {
        let j = random_range(0, len as i32) as usize;
        vec.swap(i, j);
    }
}

fn calculate_cost(route: &[usize], cities: &[City]) -> f64 {
    let mut total_cost = 0.0;
    for i in 0..route.len() - 1 {
        let city_a = &cities[route[i]];
        let city_b = &cities[route[i + 1]];
        total_cost += (((city_a.x - city_b.x).pow(2) + (city_a.y - city_b.y).pow(2)) as f64).sqrt();
    }
    // Return to the starting city
    let start_city = &cities[route[0]];
    let end_city = &cities[route[route.len() - 1]];
    total_cost += (((start_city.x - end_city.x).pow(2) + (start_city.y - end_city.y).pow(2)) as f64).sqrt();
    total_cost
}

fn initialize_particles(cities: &[City]) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(NUM_PARTICLES);
    for _ in 0..NUM_PARTICLES {
        let mut position: Vec<usize> = (0..NUM_CITIES).collect();
        shuffle_vec(&mut position);
        let cost = calculate_cost(&position, cities);
        particles.push(Particle::new(position, cost));
    }
    particles
}

fn apply_mutation_and_gaussian(position: &mut Vec<usize>) {
    // Mutation
    if random_range(0, 100) as f64 / 100.0 < MUTATION_RATE {
        let index1 = random_range(0, NUM_CITIES as i32) as usize;
        let index2 = random_range(0, NUM_CITIES as i32) as usize;
        position.swap(index1, index2);
    }

    // Simple Gaussian-like perturbation using random swap
    if random_range(0, 100) as f64 / 100.0 < MUTATION_RATE {
        let index1 = random_range(0, NUM_CITIES as i32) as usize;
        let index2 = random_range(0, NUM_CITIES as i32) as usize;
        position.swap(index1, index2);
    }
}

fn prune_particles(swarm: &mut Vec<Particle>, cities: &[City]) {
    // Sort swarm by cost in ascending order
    swarm.sort_by(|a, b| a.cost.partial_cmp(&b.cost).unwrap());

    // Remove the worst 10% of particles
    let prune_count = NUM_PARTICLES * PRUNE_PERCENTAGE / 100;
    for i in 0..prune_count {
        let particle = &mut swarm[NUM_PARTICLES - 1 - i];
        particle.position = (0..NUM_CITIES).collect();
        shuffle_vec(&mut particle.position);
        particle.cost = calculate_cost(&particle.position, cities);
        particle.best_position = particle.position.clone();
        particle.best_cost = particle.cost;
    }
}

fn update_particles(
    swarm: &mut Vec<Particle>,
    global_best_position: &mut Vec<usize>,
    global_best_cost: &mut f64,
    cities: &[City],
) {
    for particle in swarm.iter_mut() {
        // Shuffle starting positions to encourage exploration
        shuffle_vec(&mut particle.position);

        // Update based on personal best and global best
        for i in 0..NUM_CITIES {
            if random_range(0, 100) as f64 / 100.0 < COGNITIVE_COMPONENT {
                particle.position.swap(i, particle.best_position[i]);
            }
            if random_range(0, 100) as f64 / 100.0 < SOCIAL_COMPONENT {
                particle.position.swap(i, global_best_position[i]);
            }
        }

        // Apply mutation and simple Gaussian-like perturbation
        apply_mutation_and_gaussian(&mut particle.position);

        particle.cost = calculate_cost(&particle.position, cities);

        // Aggressively reward the particle if it finds a better solution
        if particle.cost < particle.best_cost {
            particle.best_cost = particle.cost;
            particle.best_position = particle.position.clone();
        }

        // Update global best if needed
        if particle.cost < *global_best_cost {
            *global_best_cost = particle.cost;
            *global_best_position = particle.position.clone();
        }
    }

    // Prune the worst-performing particles
    prune_particles(swarm, cities);
}

fn main() {
    let cities = generate_cities();
    let mut swarm = initialize_particles(&cities);

    let mut global_best_position = swarm[0].best_position.clone();
    let mut global_best_cost = swarm[0].best_cost;

    for _ in 0..MAX_ITERATIONS {
        update_particles(
            &mut swarm,
            &mut global_best_position,
            &mut global_best_cost,
            &cities,
        );
    }

    // Output the best route found
    println!("Best Route: {:?}", global_best_position);
    println!("Best Cost: {}", global_best_cost);
}
