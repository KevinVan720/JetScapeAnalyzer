set (input_dir ${PROJECT_SOURCE_DIR}/src/framework/core)
set (output_dir ${PROJECT_BINARY_DIR}/corsika/framework/core)

file (MAKE_DIRECTORY ${output_dir})

add_custom_command (
  OUTPUT  ${output_dir}/GeneratedParticleProperties.inc
          ${output_dir}/GeneratedParticleClasses.inc
          ${output_dir}/particle_db.pkl
  COMMAND ${input_dir}/pdxml_reader.py ${input_dir}/ParticleData.xml
                                       ${input_dir}/NuclearData.xml
                                       ${input_dir}/ParticleClassNames.xml
  DEPENDS ${input_dir}/pdxml_reader.py
          ${input_dir}/ParticleData.xml
          ${input_dir}/NuclearData.xml
          ${input_dir}/ParticleClassNames.xml
  WORKING_DIRECTORY
          ${output_dir}
  COMMENT "Read PYTHIA8 particle data and produce C++ source code GeneratedParticle[...].inc"
  VERBATIM
  )

add_custom_target (GenParticlesHeaders
  DEPENDS ${output_dir}/GeneratedParticleProperties.inc
          ${output_dir}/GeneratedParticleClasses.inc
          ${output_dir}/particle_db.pkl
  )
add_dependencies (CORSIKA8 GenParticlesHeaders)

install (
  FILES
  ${output_dir}/GeneratedParticleProperties.inc
  ${output_dir}/GeneratedParticleClasses.inc
  DESTINATION include/corsika/framework/core
  )
