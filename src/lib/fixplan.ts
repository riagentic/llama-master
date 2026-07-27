// src/lib/fixplan.ts — how to obtain a prerequisite that is missing.
//
// Three honest outcomes, and the difference between them matters:
//
//   download — the app fetches a self-contained build into its own directory.
//              No root, no package manager, nothing left behind on uninstall.
//   package  — the OS package manager can install it. This needs root, so the
//              exact command is shown before it runs; there is no silent
//              privileged action.
//   script   — a real, documented multi-step procedure (ROCm adds an APT
//              repository and a kernel driver). Every step is shown in full
//              before anything runs, with a link to the vendor's own page.
//   manual   — nobody should automate this. The app says what to do instead.
//
// Pure: an id, an OS and a package manager in; a plan out. The cell executes.

export type PackageManager = "apt" | "dnf" | "pacman" | "zypper" | "brew";

/** One step of a scripted install: a shell line, shown verbatim before it runs.
 *  Shell rather than argv because the vendor's own procedure uses pipes and
 *  redirection, and rewriting it would make it something other than the
 *  documented steps. */
export type FixStep = { label: string; sh: string };

export type FixPlan =
  | { kind: "download"; label: string }
  | {
    kind: "script";
    title: string;
    steps: FixStep[];
    /** The vendor page these steps come from, so they can be checked. */
    docsUrl: string;
    needsRoot: boolean;
    /** True when the machine has to restart before the tool works. */
    rebootAfter: boolean;
  }
  | {
    kind: "package";
    manager: PackageManager;
    packages: string[];
    /** Exactly what will run, root prefix excluded. Shown before it runs. */
    command: string[];
    needsRoot: boolean;
  }
  | {
    kind: "manual";
    reason: string;
    /** The vendor's own instructions. Rendered as a link, so "we will not do
     *  this for you" always comes with where to read how. */
    docsUrl?: string;
  };

/** Package names per manager. Empty = this manager has no package for it. */
const PACKAGES: Record<string, Partial<Record<PackageManager, string[]>>> = {
  compiler: {
    apt: ["build-essential"],
    dnf: ["gcc-c++", "make"],
    pacman: ["base-devel"],
    zypper: ["gcc-c++", "make"],
    brew: [], // Xcode command line tools, not a formula
  },
  git: {
    apt: ["git"],
    dnf: ["git"],
    pacman: ["git"],
    zypper: ["git"],
    brew: ["git"],
  },
  ccache: {
    apt: ["ccache"],
    dnf: ["ccache"],
    pacman: ["ccache"],
    zypper: ["ccache"],
    brew: ["ccache"],
  },
  vulkan: {
    // glslc plus the Vulkan loader headers. NOT glslangValidator: llama.cpp
    // asks cmake for glslc only, and the "missing components" note that
    // mentions glslangValidator is informational.
    apt: ["glslc", "libvulkan-dev"],
    dnf: ["glslc", "vulkan-loader-devel"],
    pacman: ["shaderc", "vulkan-headers"],
    zypper: ["shaderc", "vulkan-devel"],
    brew: ["shaderc"],
  },
  cuda: {
    apt: ["nvidia-cuda-toolkit"],
    dnf: ["cuda-toolkit"],
    pacman: ["cuda"],
    zypper: ["cuda-toolkit"],
  },
};

/** What `/etc/os-release` said.
 *
 *  `ubuntuCodename` is the field that actually decides here: Linux Mint 22,
 *  Pop!_OS 24.04 and elementary 8 all report their own `ID` and `VERSION_ID`
 *  but set `UBUNTU_CODENAME=noble`, and AMD's noble repository is exactly what
 *  they should use. Keying on `ID=ubuntu` would tell a Mint user to go read the
 *  docs while the documented steps work verbatim on their machine. */
export type Distro = {
  id: string;
  version: string;
  /** `UBUNTU_CODENAME`, empty when this is not an Ubuntu-family system. */
  ubuntuCodename: string;
};

// AMD pins both the installer .deb and the ROCm package name to a version.
// They age; these two constants and ROCM_DOCS are the only things to bump.
// Source: https://rocm.docs.amd.com/en/latest/install/rocm.html (Ubuntu 24.04)
const ROCM_INSTALLER_DEB =
  "https://repo.radeon.com/amdgpu-install/31.40/ubuntu/noble/amdgpu-install_31.40.314000-1_all.deb";
const ROCM_PACKAGE = "amdrocm-core-devel7.14";
const ROCM_DOCS =
  "https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=graphics&os=ubuntu&ubuntu-ver=24.04";

/**
 * ROCm on Ubuntu 24.04, as AMD documents it.
 *
 * This is not a package install: it adds AMD's APT repository and their GPU
 * driver, and wants a reboot. That is exactly why the steps are listed rather
 * than hidden behind one button — but "follow the docs yourself" was not much
 * of an answer either, so the app runs them, in order, with each command on
 * screen and the vendor page one click away.
 *
 * Only the combination AMD publishes steps for is offered. Anything else says
 * so and links out, because a guessed repository URL is worse than no button.
 */
/** Packages that carry the HIP development CMake files, per manager. */
const HIP_DEV_PACKAGES: Partial<Record<PackageManager, string[]>> = {
  // Verified present in AMD's noble repository (repo.radeon.com).
  apt: ["amdrocm-core-dev"],
  dnf: ["rocm-hip-devel"],
  pacman: ["rocm-hip-sdk"],
  zypper: ["rocm-hip-devel"],
};

export function rocmPlan(
  os: string,
  distro: Distro | null,
  manager: PackageManager | null = null,
  hipccPresent = false,
): FixPlan {
  // The common half-installed case: the ROCm runtime is there (hipcc, rocminfo)
  // but not the development package, so cmake stops at
  // "does not contain the HIP runtime CMake package". Installing all of ROCm
  // again would be the wrong advice — one -dev package is the fix.
  if (hipccPresent && manager) {
    const packages = HIP_DEV_PACKAGES[manager];
    if (packages) {
      return {
        kind: "package",
        manager,
        packages,
        command: installArgv(manager, packages),
        needsRoot: manager !== "brew",
      };
    }
  }

  // AMD publishes this repository for noble; anything in the noble family can
  // use it, whatever it calls itself.
  const isNoble = os === "linux" && (distro?.ubuntuCodename === "noble" ||
    (distro?.id === "ubuntu" && distro.version.startsWith("24.04")));
  if (!isNoble) {
    return {
      kind: "manual",
      reason:
        `ROCm adds AMD's own repository and a kernel driver, and the steps differ per distribution${
          distro ? ` (this is ${distro.id} ${distro.version})` : ""
        }. AMD's instructions are linked below; follow them, then re-check.`,
      docsUrl: ROCM_DOCS,
    };
  }
  return {
    kind: "script",
    title: "Install ROCm (AMD's documented steps for Ubuntu 24.04)",
    docsUrl: ROCM_DOCS,
    needsRoot: true,
    rebootAfter: true,
    steps: [
      {
        label: "Runtime libraries ROCm links against",
        sh:
          "apt-get update && apt-get install -y libatomic1 libquadmath0 wget gpg",
      },
      {
        label: "Add AMD's signing key",
        sh:
          "mkdir --parents --mode=0755 /etc/apt/keyrings && wget -qO- https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg | gpg --dearmor > /etc/apt/keyrings/amdrocm.gpg",
      },
      {
        label: "Add the ROCm repository",
        sh:
          "echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404 stable main' > /etc/apt/sources.list.d/rocm.list && apt-get update",
      },
      {
        label: "Install AMD's driver installer",
        sh:
          `cd /tmp && wget -q ${ROCM_INSTALLER_DEB} -O amdgpu-install.deb && apt-get install -y ./amdgpu-install.deb`,
      },
      {
        label: "Install the amdgpu driver (no DKMS — uses the in-tree module)",
        sh: "amdgpu-install -y --usecase=rocm --no-dkms",
      },
      {
        label: "Install the HIP compiler and ROCm development files",
        sh: `apt-get install -y ${ROCM_PACKAGE}`,
      },
      {
        label: "Give your user access to the GPU device nodes",
        sh:
          `usermod -a -G render,video ${"${SUDO_USER:-$PKEXEC_UID_NAME:-$USER}"}`,
      },
    ],
  };
}

/** The install verb for each manager, non-interactive where that is safe. */
function installArgv(m: PackageManager, packages: string[]): string[] {
  switch (m) {
    case "apt":
      return ["apt-get", "install", "-y", ...packages];
    case "dnf":
      return ["dnf", "install", "-y", ...packages];
    case "pacman":
      return ["pacman", "-S", "--noconfirm", ...packages];
    case "zypper":
      return ["zypper", "--non-interactive", "install", ...packages];
    case "brew":
      return ["brew", "install", ...packages];
  }
}

/**
 * What would fix `id` on this machine.
 *
 * `manager` is null when none was detected — then anything that is not a
 * download becomes `manual`, because inventing a command for an unknown distro
 * is worse than saying so.
 */
export function fixPlan(
  id: string,
  os: string,
  manager: PackageManager | null,
  distro: Distro | null = null,
  /** ROCm only: hipcc is on PATH but the HIP dev package may still be absent —
   *  a very different (and much smaller) fix than installing ROCm. */
  hipccPresent = false,
): FixPlan {
  if (id === "cmake") {
    return { kind: "download", label: "Download CMake into the app directory" };
  }
  if (id === "spirv") {
    return {
      kind: "download",
      label:
        "Download SPIRV-Headers into the app directory (headers only, no root needed)",
    };
  }
  if (id === "nvidia") {
    return {
      kind: "manual",
      reason:
        "A GPU driver install picks a vendor branch and needs a reboot — use your distribution's driver tool, then re-check.",
      docsUrl: "https://www.nvidia.com/en-us/drivers/",
    };
  }
  if (id === "hip") return rocmPlan(os, distro, manager, hipccPresent);
  if (id === "deno") {
    return {
      kind: "manual",
      reason: "Deno is running llama.master right now; it cannot be missing.",
    };
  }
  if (id === "compiler" && os === "darwin") {
    return {
      kind: "manual",
      reason: "Run `xcode-select --install` to get Apple's C++ toolchain.",
    };
  }

  const byManager = PACKAGES[id];
  if (!byManager) {
    return { kind: "manual", reason: `No install is known for "${id}".` };
  }
  if (!manager) {
    return {
      kind: "manual",
      reason:
        "No supported package manager was found, so there is nothing to run. Install it the way your system expects.",
    };
  }
  const packages = byManager[manager];
  if (!packages || packages.length === 0) {
    return {
      kind: "manual",
      reason: `${manager} has no package for this — install it manually.`,
      docsUrl: id === "cuda"
        ? "https://developer.nvidia.com/cuda-downloads"
        : undefined,
    };
  }
  return {
    kind: "package",
    manager,
    packages,
    command: installArgv(manager, packages),
    // Homebrew refuses to run as root; every other manager requires it.
    needsRoot: manager !== "brew",
  };
}

/** Can the app act on this plan at all? Drives whether a Fix button appears. */
export function isFixable(plan: FixPlan): boolean {
  return plan.kind !== "manual";
}

/** Every shell line a script plan will run, for the "show me first" block. */
export function scriptPreview(plan: FixPlan): string[] {
  return plan.kind === "script"
    ? plan.steps.map((st) => `# ${st.label}\n${st.sh}`)
    : [];
}

/**
 * The full command line including elevation.
 *
 * `pkexec` is preferred over `sudo`: it prompts through the desktop's own
 * authentication agent, which is the right experience from a GUI app, and it
 * never silently inherits a cached sudo timestamp.
 */
export function elevate(
  command: string[],
  opts: { isRoot: boolean; pkexec: boolean; sudo: boolean },
): string[] | null {
  if (opts.isRoot) return command;
  if (opts.pkexec) return ["pkexec", ...command];
  if (opts.sudo) return ["sudo", "-n", ...command];
  return null;
}

/** Human-readable form of what will run, for the button's tooltip. */
export function describe(plan: FixPlan): string {
  switch (plan.kind) {
    case "download":
      return plan.label;
    case "package":
      return `Runs: ${plan.command.join(" ")}${
        plan.needsRoot ? " (asks for your password)" : ""
      }`;
    case "script":
      return `${plan.title} — ${plan.steps.length} steps, asks for your password${
        plan.rebootAfter ? ", needs a reboot afterwards" : ""
      }`;
    case "manual":
      return plan.reason;
  }
}
