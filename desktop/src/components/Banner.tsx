const ASCII_LOGO = [
  " ██╗      ██████╗  ██████╗ █████╗ ██╗          ██████╗██╗     ██╗",
  " ██║     ██╔═══██╗██╔════╝██╔══██╗██║         ██╔════╝██║     ██║",
  " ██║     ██║   ██║██║     ███████║██║  █████╗ ██║     ██║     ██║",
  " ██║     ██║   ██║██║     ██╔══██║██║  ╚════╝ ██║     ██║     ██║",
  " ███████╗╚██████╔╝╚██████╗██║  ██║███████╗    ╚██████╗███████╗██║",
  " ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝     ╚═════╝╚══════╝╚═╝",
].join("\n")

export function Banner({ version }: { version: string }) {
  return (
    <div className="banner">
      <pre className="banner-art">{ASCII_LOGO}</pre>
      <div className="banner-info">
        <span className="banner-version">v{version}</span>
        <span className="banner-desc">Local AI coding agent</span>
      </div>
    </div>
  )
}
