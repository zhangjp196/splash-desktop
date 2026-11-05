import SwiftUI

/// One row in the command palette: a page to open or a server action to run.
struct PaletteAction: Identifiable {
    let id: String
    let title: String
    let subtitle: String
    let systemImage: String
    let tint: Color
    let run: () -> Void
}

extension Notification.Name {
    /// The app menu's ⌘K, which reaches the window whichever view owns the
    /// keyboard — including the embedded chat page.
    static let showCommandPalette = Notification.Name("splash.show-command-palette")
    /// A short confirmation shown at the bottom of the window, such as a
    /// copied API address. The object is the already-localized message.
    static let splashToast = Notification.Name("splash.toast")
}

/// The ⌘K palette: a quick way to move between pages and run the server's
/// main actions without leaving the keyboard.
struct CommandPalette: View {
    let actions: [PaletteAction]
    @Binding var isPresented: Bool

    @State private var query = ""
    @State private var index = 0
    @FocusState private var focused: Bool

    private var filtered: [PaletteAction] {
        let needle = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !needle.isEmpty else { return actions }
        return actions.filter {
            $0.title.lowercased().contains(needle)
                || $0.subtitle.lowercased().contains(needle)
        }
    }

    var body: some View {
        ZStack(alignment: .top) {
            Color.black.opacity(0.44)
                .ignoresSafeArea()
                .onTapGesture { close() }

            VStack(spacing: 0) {
                HStack(spacing: 11) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(SplashPalette.cyan)
                    TextField("", text: $query, prompt: Text(L10n.string("palette.placeholder")))
                        .textFieldStyle(.plain)
                        .font(.system(size: 14, weight: .medium))
                        .focused($focused)
                        .onSubmit { runSelected() }
                        .onKeyPress(.downArrow) {
                            move(1)
                            return .handled
                        }
                        .onKeyPress(.upArrow) {
                            move(-1)
                            return .handled
                        }
                    Text(verbatim: "⌘K")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundStyle(SplashPalette.faint)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(SplashPalette.surface.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 14)

                Rectangle()
                    .fill(SplashPalette.border.opacity(0.4))
                    .frame(height: 1)

                if filtered.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 20, weight: .light))
                            .foregroundStyle(SplashPalette.faint)
                        Text(L10n.string("palette.empty"))
                            .font(.system(size: 12))
                            .foregroundStyle(SplashPalette.muted)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 34)
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            VStack(spacing: 2) {
                                ForEach(Array(filtered.enumerated()), id: \.element.id) { position, action in
                                    row(action, selected: position == index)
                                        .id(action.id)
                                        .onHover { inside in
                                            if inside { index = position }
                                        }
                                }
                            }
                            .padding(6)
                        }
                        .frame(maxHeight: 330)
                        .onChange(of: index) { _, value in
                            guard filtered.indices.contains(value) else { return }
                            withAnimation(.easeOut(duration: 0.15)) {
                                proxy.scrollTo(filtered[value].id, anchor: .center)
                            }
                        }
                    }
                }

                Rectangle()
                    .fill(SplashPalette.border.opacity(0.4))
                    .frame(height: 1)

                HStack(spacing: 12) {
                    hint("↑↓", L10n.string("palette.navigate"))
                    hint("↵", L10n.string("palette.run"))
                    hint("esc", L10n.string("palette.close"))
                    Spacer()
                    Text(verbatim: "Splash")
                        .font(.system(size: 9, weight: .semibold, design: .rounded))
                        .foregroundStyle(SplashPalette.faint)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
            }
            .frame(width: 560)
            .background {
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(.ultraThinMaterial)
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(SplashPalette.surface.opacity(0.82))
            }
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                SplashPalette.sheen.opacity(0.24),
                                SplashPalette.border.opacity(0.4),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.8
                    )
            )
            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            .shadow(color: .black.opacity(0.45), radius: 40, y: 18)
            .padding(.top, 92)
        }
        .onExitCommand { close() }
        .onMoveCommand { direction in
            switch direction {
            case .down: move(1)
            case .up: move(-1)
            default: break
            }
        }
        .defaultFocus($focused, true)
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
                focused = true
            }
        }
        .onChange(of: query) { _, _ in
            index = 0
        }
    }

    private func row(_ action: PaletteAction, selected: Bool) -> some View {
        Button {
            run(action)
        } label: {
            HStack(spacing: 11) {
                Image(systemName: action.systemImage)
                    .font(.system(size: 11.5, weight: .bold))
                    .foregroundStyle(action.tint)
                    .frame(width: 28, height: 28)
                    .background(action.tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(action.tint.opacity(0.24), lineWidth: 0.7)
                    )
                VStack(alignment: .leading, spacing: 1) {
                    Text(action.title)
                        .font(.system(size: 12.5, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    Text(action.subtitle)
                        .font(.system(size: 9.5))
                        .foregroundStyle(SplashPalette.faint)
                        .lineLimit(1)
                }
                Spacer(minLength: 6)
                if selected {
                    Text(verbatim: "↵")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(SplashPalette.cyan)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                selected ? action.tint.opacity(0.13) : Color.clear,
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private func hint(_ key: String, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(key)
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(SplashPalette.muted)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(SplashPalette.surface.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(SplashPalette.faint)
        }
    }

    private func move(_ delta: Int) {
        guard !filtered.isEmpty else { return }
        index = (index + delta + filtered.count) % filtered.count
    }

    private func runSelected() {
        guard filtered.indices.contains(index) else { return }
        run(filtered[index])
    }

    private func run(_ action: PaletteAction) {
        close()
        action.run()
    }

    private func close() {
        isPresented = false
    }
}
