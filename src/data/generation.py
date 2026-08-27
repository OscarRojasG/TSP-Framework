import os
import h5py
from concurrent.futures import ProcessPoolExecutor
from settings import DATA_FOLDER
from instances.instances import read_instances
from solvers.ortools import solve
from TSP import TSPState

# Variables globales para los workers
worker_input_adapter = None
worker_output_adapter = None

def init_worker(input_adapter_config, output_adapter_config):
    """Inicializa los adaptadores globalmente en cada proceso worker."""
    global worker_input_adapter
    global worker_output_adapter

    in_class, *in_args = input_adapter_config
    out_class, *out_args = output_adapter_config

    worker_input_adapter = in_class(*in_args)
    worker_output_adapter = out_class(*out_args)

def generate_data_from_instance(instance):
    """Resuelve una instancia de TSP y extrae los vectores de cada estado."""
    solution = solve(instance)
    
    # Si la instancia no se pudo resolver, retornamos None
    if not solution or not solution.tour:
        return None

    state = TSPState(instance)
    input_vecs = []
    output_vecs = []

    # Iteramos sobre el tour (omitiendo el inicio y fin si es necesario)
    for city in solution.tour[1:-1]:
        # Usamos los adaptadores abstractos inicializados en el worker
        input_vec = worker_input_adapter.input_2_vec(state)
        output_vec = worker_output_adapter.output_2_vec(state, city)

        input_vecs.append(input_vec)
        output_vecs.append(output_vec)

        # Actualizamos el estado visitando la ciudad
        state.visit_city(city)

    return input_vecs, output_vecs

def generate_data(instances, input_adapter_config, output_adapter_config, num_workers):
    """Ejecuta la generación de datos en paralelo para una lista de instancias."""
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(input_adapter_config, output_adapter_config)
    ) as executor:
        results = list(executor.map(generate_data_from_instance, instances))

    # Inicializamos adaptadores en el proceso principal para recolectar resultados
    in_class, *in_args = input_adapter_config
    out_class, *out_args = output_adapter_config
    main_input_adapter = in_class(*in_args)
    main_output_adapter = out_class(*out_args)

    # Agregamos los resultados generados por los workers
    for result in results:
        if result is None:
            continue
        
        input_vecs, output_vecs = result
        for in_vec, out_vec in zip(input_vecs, output_vecs):
            main_input_adapter.add(in_vec)
            main_output_adapter.add(out_vec)

    # Obtenemos los diccionarios de datos finales
    input_data = main_input_adapter.get()
    output_data = main_output_adapter.get()

    return input_data, output_data

def save_data(input_data, output_data, filename, verbose=True):
    """Guarda los diccionarios de datos en formato HDF5."""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    output_path = DATA_FOLDER / filename

    with h5py.File(output_path, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")

        input_keys = list(input_data.keys())
        for key in input_keys:
            g_input.create_dataset(key, data=input_data[key])
        g_input.attrs['key_order'] = input_keys

        output_keys = list(output_data.keys())
        for key in output_keys:
            g_output.create_dataset(key, data=output_data[key])
        g_output.attrs['key_order'] = output_keys

    # Imprimimos el tamaño usando una de las llaves como referencia
    ref_key = input_keys[0]

    if verbose:
        print(f"Datos guardados en: {output_path} (Tamaño: {len(input_data[ref_key])})")

def generate_train_data(instance_file, data_filename, input_adapter_config, output_adapter_config, num_workers=4, size=10000):
    """Lee el archivo, orquesta la generación paralela y guarda limitando al 'size'."""
    instances = read_instances(instance_file)
    
    input_data, output_data = generate_data(
        instances, 
        input_adapter_config, 
        output_adapter_config, 
        num_workers
    )

    # Truncamiento de datos para respetar el parámetro 'size' original
    if size is not None:
        for k in input_data:
            input_data[k] = input_data[k][:size]
        for k in output_data:
            output_data[k] = output_data[k][:size]

    save_data(input_data, output_data, data_filename)