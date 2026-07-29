import glob
import re
import tables

# Directorio donde están los chis_RunXXXX.h5
input_path = "/data/cta/users-ifae/summer_students/agl/try_2/chis_Run*.h5"

# Archivo de salida
output_file = "chis_all.h5"

# Buscar todos los archivos
files = sorted(glob.glob(input_path))

# Crear archivo de salida
h5_out = tables.open_file(output_file, mode="w")

group = h5_out.create_group("/", "chis2", "All chi2 results")


class ChiSquared(tables.IsDescription):
    run = tables.Int32Col()
    check_index = tables.Int32Col()
    dof = tables.Int32Col()
    chi2 = tables.Float64Col()


table = h5_out.create_table(
    group,
    "results",
    ChiSquared,
    "Combined chi2 results"
)

row_out = table.row

for filename in files:

    # Extraer el RunID del nombre del archivo
    match = re.search(r"Run(\d+)", filename)
    if match is None:
        print(f"Skipping {filename}")
        continue

    run = int(match.group(1))

    print(f"Processing Run {run}")

    h5 = tables.open_file(filename)

    data = h5.root.chis2.results

    for r in data:

        row_out["run"] = run
        row_out["check_index"] = r["check_index"]
        row_out["dof"] = r["dof"]
        row_out["chi2"] = r["chi2"]

        row_out.append()

    h5.close()

table.flush()
h5_out.close()

print(f"\nCreated {output_file}")