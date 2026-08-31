### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 22067896-42cf-11f1-9da8-ef5d3fc30ce3
begin
	using LinearAlgebra
	using PlutoUI 
	using Statistics
	using PyCall # Run python inside julia. It will use in intrpolation.
	
	md"""
	# Package
	"""
end

# ╔═╡ 3efb62fe-213e-4d93-adf1-740ce6525977
begin
    if !isdefined(@__MODULE__, :nonKam) # check module is already imported or not.
    include("src/nonKam.jl") # function `clf_BW_m_ref_circuit`
    using .nonKam
end
end

# ╔═╡ 0242125d-609c-4700-aff9-a3ead51a2796
md"""
# Parameters
"""

# ╔═╡ f81a22f7-2db1-4246-b38c-41965aedb064
md"""
## KHO parameters
"""

# ╔═╡ 9953ab45-0265-41a0-8766-2bc854f588a3
begin
	# KHO parameters
	R = 5
	ω = 2*π/R
	τ = 1
	K = 1.5
	nothing
end

# ╔═╡ 605cad60-36ce-48ae-83a3-efe329c3ca66
md"""
## Simulation parameters
"""

# ╔═╡ 4bd2fe4b-e8ca-42aa-99d3-449e048d504e
begin
	N = 500          # Hilbert space dimension
	t_max = 100      # Maximum time steps for QFI calculation
	nothing
end

# ╔═╡ 865622c4-cdc1-4dcd-96c8-8b45da2d5485
md"""
## Initial state
"""

# ╔═╡ cce98700-9f19-4c74-946a-1446b4abf062
begin
	# Initial state (ground state)
	ψ0 = zeros(ComplexF64, N)
	ψ0[1] = 1.0 + 0im  # |0⟩ state
	nothing
end

# ╔═╡ 3ef1ebcb-cb49-4c29-9d5d-4338febfdc71
md"""
# Plots
"""

# ╔═╡ 0878408f-c15c-4b51-aaaf-9c65b8a4f934


# ╔═╡ 100b8828-8548-49ee-b24e-48787b6b5ce7
# plt.close("all")

# ╔═╡ 049819db-82c0-4ea7-9ba2-3502cc835d8e
md"""
# Operators
"""

# ╔═╡ 237ce804-65f9-4d04-a67f-498307f23c30
begin
	# ladder operators and Floquet unitary
	a = diagm(1 => sqrt.(1:N-1))
	adag = a'
	U = KickedHO_floquet(N, ω, τ, K)
	nothing
end

# ╔═╡ e1bb82c9-3720-400b-8fb0-54fb18511e6a
md"""
# Time evolution
"""

# ╔═╡ 43119d43-42c0-46cd-9a0b-3067472d84f1
begin
	# Calculate QFI over time
	qfi_array = qfi_with_time_provided_U_a(U, a, adag, ω, τ, K, t_max, ψ0)
	nothing
end

# ╔═╡ 0ae9903b-5343-461e-b26e-b2be4b9db0b3
t_array = collect(1:t_max);

# ╔═╡ c2a373ad-4b76-4f46-8637-f2a0da32da6d
begin
	# Import matplotlib.pyplot as plt
	plt = pyimport("matplotlib.pyplot")
	mpl = pyimport("matplotlib")
	np = pyimport("numpy")
	md"""
	# Pyhon library
	"""
end

# ╔═╡ 858e4c32-e1c5-4cfe-b557-aec42c1621c9
begin
    # 1. Use subplots() with no arguments for a single plot
    # This gives you one 'fig' and one 'ax'
    fig, ax = plt.subplots(figsize=(6, 5))
	ax.tick_params(axis="both", labelsize=18)

    # 2. Plot your data
    ax.plot(t_array, qfi_array, label="name", marker="s")
	ax.plot(collect(2:t_max),collect(2:t_max).^3)
    
    # Add labels (optional but good practice)
    # ax.set_title("Single Plot: ta^2")
    ax.set_xlabel("\$t\$", fontsize=18)
    ax.set_ylabel("\$I(\\left|\\psi\\right>;R)\$", fontsize=18)
	ax.set_xscale("log")
	ax.set_yscale("log")
    ax.grid(false) # Adds a background grid
    ax.legend()

    # 3. Save
    mkpath("plot")
    fig.savefig("plot/my_plot.pdf", bbox_inches="tight")

    # 4. Clean up memory
    plt.close(fig)
    
    "Single plot saved successfully to plot/my_plot.pdf"
end

# ╔═╡ 4f803692-f76f-4af9-ad2d-9dcef3e6d026
begin
	ta = np.array(collect(1:10))
end

# ╔═╡ 815c4978-7d91-4b81-abe4-a799278b0097
TableOfContents(title="nonKam")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoUI = "~0.7.72"
PyCall = "~1.96.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "e6f12b412e9d14d5ac40e682be1e54ef127cfd58"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "8f06b0cfa4c514c7b9546756dbae91fcfbc92dc9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "f53232a27a8c1c836d3998ae1e17d898d4df2a46"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.72"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─0242125d-609c-4700-aff9-a3ead51a2796
# ╟─f81a22f7-2db1-4246-b38c-41965aedb064
# ╠═9953ab45-0265-41a0-8766-2bc854f588a3
# ╟─605cad60-36ce-48ae-83a3-efe329c3ca66
# ╠═4bd2fe4b-e8ca-42aa-99d3-449e048d504e
# ╟─865622c4-cdc1-4dcd-96c8-8b45da2d5485
# ╠═cce98700-9f19-4c74-946a-1446b4abf062
# ╟─3ef1ebcb-cb49-4c29-9d5d-4338febfdc71
# ╠═0878408f-c15c-4b51-aaaf-9c65b8a4f934
# ╠═858e4c32-e1c5-4cfe-b557-aec42c1621c9
# ╠═4f803692-f76f-4af9-ad2d-9dcef3e6d026
# ╠═100b8828-8548-49ee-b24e-48787b6b5ce7
# ╟─049819db-82c0-4ea7-9ba2-3502cc835d8e
# ╠═237ce804-65f9-4d04-a67f-498307f23c30
# ╟─e1bb82c9-3720-400b-8fb0-54fb18511e6a
# ╠═43119d43-42c0-46cd-9a0b-3067472d84f1
# ╠═0ae9903b-5343-461e-b26e-b2be4b9db0b3
# ╠═c2a373ad-4b76-4f46-8637-f2a0da32da6d
# ╠═22067896-42cf-11f1-9da8-ef5d3fc30ce3
# ╠═815c4978-7d91-4b81-abe4-a799278b0097
# ╠═3efb62fe-213e-4d93-adf1-740ce6525977
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
