add_library(macquerel SHARED
    simulator.py
)

target_link_libraries(macquerel PRIVATE Python::Python)