#!/usr/bin/env php
<?php
// phpisa_dump.php -- run a PAL program on phpisa and dump the final machine
// state as JSON (registers, memory = program words, pc, br, dir, output).
//
// phpisa itself is not modified: this driver only `require`s its src/runner.php
// and calls phpisa_run().  Used by tools/difftest_pisa.py.
//
// usage: php tools/phpisa_dump.php [--phpisa DIR] [--max-steps N] [--regs JSON] file.pal
//   --phpisa DIR   phpisa checkout (default: $PHPISA_DIR or ../../phpisa)
//   --max-steps N  step budget (phpisa's default is 1,000,000)
//   --regs JSON    initial registers, e.g. '{"R1":5,"R2":0}'
// Output (one JSON object): {"ok":true, "finished":bool, "pc":int, "br":int,
//   "dir":int, "steps_exceeded":bool, "registers":{"R0":..}, "mem":{"0":..},
//   "symbols":{"LABEL":idx}, "program_length":int, "output":"...", "error":null}

ini_set('display_errors', 'stderr');
error_reporting(E_ALL);

$args     = array_slice($argv, 1);
$dir      = getenv('PHPISA_DIR') ?: (__DIR__ . '/../../phpisa');
$maxSteps = 1000000;
$regs     = [];
$file     = null;
for ($i = 0; $i < count($args); $i++) {
	switch ($args[$i]) {
		case '--phpisa':    $dir = $args[++$i]; break;
		case '--max-steps': $maxSteps = (int)$args[++$i]; break;
		case '--regs':      $regs = json_decode($args[++$i], true) ?? []; break;
		default:            $file = $args[$i];
	}
}
if ($file === null) {
	fwrite(STDERR, "usage: phpisa_dump.php [--phpisa DIR] [--max-steps N] [--regs JSON] file.pal\n");
	exit(2);
}
$runner = rtrim($dir, '/') . '/src/runner.php';
if (!is_readable($runner)) {
	echo json_encode(['ok' => false, 'error' => "cannot find phpisa runner at $runner"]), "\n";
	exit(1);
}
require_once $runner;

$code = file_get_contents($file);
$result = null;
$error  = null;
$exceeded = false;
// compile() calls exit(1)/die() on a bad mnemonic or unknown label; we let
// that propagate (the caller sees a non-JSON stdout/stderr and reports it).
try {
	$result = phpisa_run($code, [
		'capture'           => true,
		'max_steps'         => $maxSteps,
		'initial_registers' => $regs,
	]);
} catch (RuntimeException $e) {
	$exceeded = str_contains($e->getMessage(), 'max_steps');
	$error    = $e->getMessage();
	// runner does not clean its output buffer on this path
	while (ob_get_level() > 0) $out = ob_get_clean();
	global $registers, $PC, $DIRECTION, $BRANCH_REG;
	$result = [
		'registers'  => $registers,
		'output'     => $out ?? '',
		'pc'         => $PC,
		'direction'  => $DIRECTION,
		'branch_reg' => $BRANCH_REG,
		'finished'   => false,
	];
}

global $program, $SYMBOLS;
$mem = [];
foreach ($program as $addr => $word) {
	$mem[(string)$addr] = is_string($word) ? (int)preg_replace('/(.*:\s+)?DATA(\s+)/', '', $word) : $word;
}
echo json_encode([
	'ok'             => $error === null,
	'error'          => $error,
	'steps_exceeded' => $exceeded,
	'finished'       => $result['finished'],
	'pc'             => $result['pc'],
	'br'             => $result['branch_reg'],
	'dir'            => $result['direction'],
	'registers'      => $result['registers'],
	'mem'            => $mem,
	'symbols'        => $SYMBOLS,
	'program_length' => count($program),
	'output'         => $result['output'],
], JSON_FORCE_OBJECT), "\n";
