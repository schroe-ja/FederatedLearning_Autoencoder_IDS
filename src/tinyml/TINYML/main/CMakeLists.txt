idf_component_register(
    SRCS "TinyML_main.cpp"
    INCLUDE_DIRS "."
    REQUIRES 
        espressif__esp-tflite-micro
        esp_timer
        driver
        log
        freertos
        esp_system
        esp_rom
        esp_common
        hal
        soc
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
# Fix for TFLite Shadow Warnings
# This must come AFTER idf_component_register
target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-error=shadow)
# --- TFLITE WORKAROUND ---
# Zwinge die TFLite-Komponente von außen auf -O2 und ignoriere ihre Warnungen.
# GCC nutzt immer das *letzte* -O Flag. Wenn TFLite intern -O3 setzt, 
# hängen wir -O2 einfach hinten dran, damit es überschrieben wird.
target_compile_options(__idf_espressif__esp-tflite-micro PRIVATE -O2 -Wno-error=shadow)