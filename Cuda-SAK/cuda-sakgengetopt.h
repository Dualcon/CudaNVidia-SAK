/** @file cuda-sakgengetopt.h
 *  @brief The header file for the command line option parser
 *  generated by GNU Gengetopt version 2.22.6
 *  http://www.gnu.org/software/gengetopt.
 *  DO NOT modify this file, since it can be overwritten
 *  @author GNU Gengetopt by Lorenzo Bettini */

#ifndef CUDA_SAKGENGETOPT_H
#define CUDA_SAKGENGETOPT_H

/* If we use autoconf.  */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h> /* for FILE */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef CMDLINE_PARSER_PACKAGE
/** @brief the program name (used for printing errors) */
#define CMDLINE_PARSER_PACKAGE "mei-team_cuda-sak"
#endif

#ifndef CMDLINE_PARSER_PACKAGE_NAME
/** @brief the complete program name (used for help and version) */
#define CMDLINE_PARSER_PACKAGE_NAME "mei-team_cuda-sak"
#endif

#ifndef CMDLINE_PARSER_VERSION
/** @brief the program version */
#define CMDLINE_PARSER_VERSION "1.0"
#endif

/** @brief Where the command line options are stored */
struct gengetopt_args_info
{
  const char *help_help; /**< @brief Print help and exit help description.  */
  const char *version_help; /**< @brief Print version and exit help description.  */
  const char *device_help; /**< @brief opção -d / -device help description.  */
  const char *handleerror_help; /**< @brief Generates the CUDA error processing code help description.  */
  char * callkernel_arg;	/**< @brief Generates the code that calls kernel. Requires the name of the kernel as paremeter.  */
  char * callkernel_orig;	/**< @brief Generates the code that calls kernel. Requires the name of the kernel as paremeter original value given at command line.  */
  const char *callkernel_help; /**< @brief Generates the code that calls kernel. Requires the name of the kernel as paremeter help description.  */
  int genmaps_arg;	/**< @brief Generates the mapping for n-D geometries to 1-D kernel. n should be given and should be between 1 and 3.  */
  char * genmaps_orig;	/**< @brief Generates the mapping for n-D geometries to 1-D kernel. n should be given and should be between 1 and 3 original value given at command line.  */
  const char *genmaps_help; /**< @brief Generates the mapping for n-D geometries to 1-D kernel. n should be given and should be between 1 and 3 help description.  */
  const char *occupancy_help; /**< @brief Generates the skeleton code that uses the CUDA 6.5 occupancy API help description.  */
  int preprocessor_arg;	/**< @brief Generates the wrappes for preprocessor. Requires architecture as parameter, which must be a number greater than 1 and a multiple of 10..  */
  char * preprocessor_orig;	/**< @brief Generates the wrappes for preprocessor. Requires architecture as parameter, which must be a number greater than 1 and a multiple of 10. original value given at command line.  */
  const char *preprocessor_help; /**< @brief Generates the wrappes for preprocessor. Requires architecture as parameter, which must be a number greater than 1 and a multiple of 10. help description.  */
  const char *reduction_help; /**< @brief Generates the skeleton kernel code regarding the traditional log2 N reduction algorithm. help description.  */
  char * skeleton_arg;	/**< @brief Generates the source code if a CUDA kernel skeleton.  */
  char * skeleton_orig;	/**< @brief Generates the source code if a CUDA kernel skeleton original value given at command line.  */
  const char *skeleton_help; /**< @brief Generates the source code if a CUDA kernel skeleton help description.  */
  const char *timing_help; /**< @brief Generates the source code used for timing the execution of a CUDA kernel. help description.  */
  const char *unified_help; /**< @brief Generates the skeleton code that uses CUDA 6.5 unified memory API. help description.  */
  const char *about_help; /**< @brief Displays the credits of the application. help description.  */
  const char *testall_help; /**< @brief Activates all the options that generate code. help description.  */
  char * filename_arg;	/**< @brief Needs filename and writes output to that file.  */
  char * filename_orig;	/**< @brief Needs filename and writes output to that file original value given at command line.  */
  const char *filename_help; /**< @brief Needs filename and writes output to that file help description.  */
  char * students_arg;	/**< @brief Used along with other option(to define the output).Needs origin filename and line to write in..  */
  char * students_orig;	/**< @brief Used along with other option(to define the output).Needs origin filename and line to write in. original value given at command line.  */
  const char *students_help; /**< @brief Used along with other option(to define the output).Needs origin filename and line to write in. help description.  */
  
  unsigned int help_given ;	/**< @brief Whether help was given.  */
  unsigned int version_given ;	/**< @brief Whether version was given.  */
  unsigned int device_given ;	/**< @brief Whether device was given.  */
  unsigned int handleerror_given ;	/**< @brief Whether handleerror was given.  */
  unsigned int callkernel_given ;	/**< @brief Whether callkernel was given.  */
  unsigned int genmaps_given ;	/**< @brief Whether genmaps was given.  */
  unsigned int occupancy_given ;	/**< @brief Whether occupancy was given.  */
  unsigned int preprocessor_given ;	/**< @brief Whether preprocessor was given.  */
  unsigned int reduction_given ;	/**< @brief Whether reduction was given.  */
  unsigned int skeleton_given ;	/**< @brief Whether skeleton was given.  */
  unsigned int timing_given ;	/**< @brief Whether timing was given.  */
  unsigned int unified_given ;	/**< @brief Whether unified was given.  */
  unsigned int about_given ;	/**< @brief Whether about was given.  */
  unsigned int testall_given ;	/**< @brief Whether testall was given.  */
  unsigned int filename_given ;	/**< @brief Whether filename was given.  */
  unsigned int students_given ;	/**< @brief Whether students was given.  */

  int grupo_principal_group_counter; /**< @brief Counter for group grupo_principal */
} ;

/** @brief The additional parameters to pass to parser functions */
struct cmdline_parser_params
{
  int override; /**< @brief whether to override possibly already present options (default 0) */
  int initialize; /**< @brief whether to initialize the option structure gengetopt_args_info (default 1) */
  int check_required; /**< @brief whether to check that all required options were provided (default 1) */
  int check_ambiguity; /**< @brief whether to check for options already specified in the option structure gengetopt_args_info (default 0) */
  int print_errors; /**< @brief whether getopt_long should print an error message for a bad option (default 1) */
} ;

/** @brief the purpose string of the program */
extern const char *gengetopt_args_info_purpose;
/** @brief the usage string of the program */
extern const char *gengetopt_args_info_usage;
/** @brief the description string of the program */
extern const char *gengetopt_args_info_description;
/** @brief all the lines making the help output */
extern const char *gengetopt_args_info_help[];

/**
 * The command line parser
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser (int argc, char **argv,
  struct gengetopt_args_info *args_info);

/**
 * The command line parser (version with additional parameters - deprecated)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param override whether to override possibly already present options
 * @param initialize whether to initialize the option structure my_args_info
 * @param check_required whether to check that all required options were provided
 * @return 0 if everything went fine, NON 0 if an error took place
 * @deprecated use cmdline_parser_ext() instead
 */
int cmdline_parser2 (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  int override, int initialize, int check_required);

/**
 * The command line parser (version with additional parameters)
 * @param argc the number of command line options
 * @param argv the command line options
 * @param args_info the structure where option information will be stored
 * @param params additional parameters for the parser
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_ext (int argc, char **argv,
  struct gengetopt_args_info *args_info,
  struct cmdline_parser_params *params);

/**
 * Save the contents of the option struct into an already open FILE stream.
 * @param outfile the stream where to dump options
 * @param args_info the option struct to dump
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_dump(FILE *outfile,
  struct gengetopt_args_info *args_info);

/**
 * Save the contents of the option struct into a (text) file.
 * This file can be read by the config file parser (if generated by gengetopt)
 * @param filename the file where to save
 * @param args_info the option struct to save
 * @return 0 if everything went fine, NON 0 if an error took place
 */
int cmdline_parser_file_save(const char *filename,
  struct gengetopt_args_info *args_info);

/**
 * Print the help
 */
void cmdline_parser_print_help(void);
/**
 * Print the version
 */
void cmdline_parser_print_version(void);

/**
 * Initializes all the fields a cmdline_parser_params structure 
 * to their default values
 * @param params the structure to initialize
 */
void cmdline_parser_params_init(struct cmdline_parser_params *params);

/**
 * Allocates dynamically a cmdline_parser_params structure and initializes
 * all its fields to their default values
 * @return the created and initialized cmdline_parser_params structure
 */
struct cmdline_parser_params *cmdline_parser_params_create(void);

/**
 * Initializes the passed gengetopt_args_info structure's fields
 * (also set default values for options that have a default)
 * @param args_info the structure to initialize
 */
void cmdline_parser_init (struct gengetopt_args_info *args_info);
/**
 * Deallocates the string fields of the gengetopt_args_info structure
 * (but does not deallocate the structure itself)
 * @param args_info the structure to deallocate
 */
void cmdline_parser_free (struct gengetopt_args_info *args_info);

/**
 * Checks that all the required options were specified
 * @param args_info the structure to check
 * @param prog_name the name of the program that will be used to print
 *   possible errors
 * @return
 */
int cmdline_parser_required (struct gengetopt_args_info *args_info,
  const char *prog_name);

extern const char *cmdline_parser_genmaps_values[];  /**< @brief Possible values for genmaps. */


#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* CUDA_SAKGENGETOPT_H */